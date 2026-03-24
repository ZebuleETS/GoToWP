"""
Integration multi-UAV de votre simulation de planeur avec PX4 SITL
Support de plusieurs drones avec thermiques partagées
"""

import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw, AttitudeRate
from mavsdk.telemetry import LandedState, FlightMode
from mavsdk.action import OrbitYawBehavior
import time
import pymap3d as pm
from GoToWP import gotoWaypointMulti
from trajectory import TrajectoryEvaluator, generate_all_trajectories, generate_random_obstacles, LawnMowerTrajectory
from thermal import ThermalMap, detect_thermal_at_position
from thermal_ros_bridge import ThermalROSBridge
from compute import (convert_cylindrical_obstacles_to_polygons, find_nearest_waypoint, 
                     get_sink_rate, calculate_optimal_soaring_parameters)
from Scenario import (TestScenario, PerformanceMetrics, SurveillanceObject, 
                      ScenarioGenerator, PerformanceAnalyzer, select_scenario)


# ─── Constantes atmosphériques (ISA) ────────────────────────────────────
_ACC_SEA_LEVEL    = 9.80665
_MEAN_EARTH_RADIUS = 6371009
_T_SEA_LEVEL      = 288.15
_RHO_SEA_LEVEL    = 1.225
_TROPO_LAPSE_RATE = -0.0065
_R_AIR            = 287.058


def _compute_atmosphere(working_floor: float) -> dict:
    """
    Calcule la gravité et la densité de l'air à l'altitude *working_floor*
    selon le modèle ISA (International Standard Atmosphere).

    Returns:
        dict: {'grav': float, 'air_density': float}
    """
    grav = _ACC_SEA_LEVEL * (_MEAN_EARTH_RADIUS / (_MEAN_EARTH_RADIUS + working_floor))
    T_fin = _T_SEA_LEVEL + _TROPO_LAPSE_RATE * working_floor
    air_density = _RHO_SEA_LEVEL * (T_fin / _T_SEA_LEVEL) ** (
        -grav / (_TROPO_LAPSE_RATE * _R_AIR) - 1
    )
    return {'grav': grav, 'air_density': air_density}


def _remove_evaluation_obstacles(params, uav_id, thermal_id):
    """Retire les obstacles de type evaluation_zone pour un UAV/thermique donné."""
    params['obstacles'] = [
        obs for obs in params['obstacles']
        if not (obs.get('type') == 'evaluation_zone'
                and obs.get('uav_id') == uav_id
                and obs.get('thermal_id') == thermal_id)
    ]


def _cancel_seeking(FLT_track_u, GOAL_WPs_u, current_wp_indices, u):
    """
    Annule le seeking en cours pour un UAV :
    retire le waypoint de secours inséré dans GOAL_WPs (s'il n'a pas encore été
    atteint) et réinitialise les flags seeking dans FLT_track.
    """
    wp_idx = FLT_track_u.get('seeking_wp_idx')
    if wp_idx is not None and wp_idx < len(GOAL_WPs_u['X']):
        # Ne retirer le WP que si le drone ne l'a pas encore dépassé
        if current_wp_indices[u] <= wp_idx:
            GOAL_WPs_u['X'].pop(wp_idx)
            GOAL_WPs_u['Y'].pop(wp_idx)
            GOAL_WPs_u['Z'].pop(wp_idx)
    FLT_track_u['seeking_thermal'] = False
    FLT_track_u['seeking_thermal_id'] = None
    FLT_track_u['seeking_wp_idx'] = None


def _check_thermal_altitude_conflict(uav_id, thermal_id, uav_altitude, controller, FLT_track, nUAVs, min_alt_separation=80.0):
    """
    Vérifie si un drone peut entrer dans une thermique sans conflit d'altitude
    avec les autres drones déjà présents.
    
    Plusieurs drones peuvent utiliser la même thermique s'ils sont
    suffisamment espacés en altitude (>= min_alt_separation).
    
    Args:
        uav_id: ID du drone candidat
        thermal_id: ID de la thermique visée
        uav_altitude: Altitude actuelle du drone candidat (m)
        controller: MultiUAVController
        FLT_track: Données de vol
        nUAVs: Nombre total de drones
        min_alt_separation: Séparation minimale en altitude (m), défaut 80m
        
    Returns:
        bool: True s'il y a un conflit (trop proche en altitude), False sinon
    """
    for other_u in range(nUAVs):
        if other_u == uav_id:
            continue
        bridge = controller.bridges[other_u]
        # Seuls les drones en orbite dans cette même thermique posent problème
        if not bridge.is_orbiting or bridge.orbit_thermal_id != thermal_id:
            continue
        # Récupérer l'altitude de l'autre drone
        if FLT_track[other_u]['Z']:
            other_alt = FLT_track[other_u]['Z'][-1]
            alt_diff = abs(uav_altitude - other_alt)
            if alt_diff < min_alt_separation:
                return True  # Conflit : trop proche en altitude
    return False  # Pas de conflit


class PX4SITLBridge:
    """
    Pont entre votre simulation et PX4 SITL pour un seul UAV
    """
    
    def __init__(self, uav_id, params, UAV_data, connection_port, mavsdk_port):
        self.uav_id = uav_id
        self.params = params
        self.UAV_data = UAV_data
        self.connection_port = connection_port
        self.mavsdk_port = mavsdk_port
        self.drone = None
        self.is_connected = False
        self.home_position = None
        self.simulation_origin = None
        self.own_home = None  # Home propre du drone (pour commandes NED PX4)
        
        # Taux de mise à jour (Hz)
        self.update_rate = 10
        
        # Keepalive pour offboard
        self.offboard_keepalive_task = None
        self.last_position_ned = PositionNedYaw(0.0, 0.0, 0.0, 0.0)
        self.last_velocity_ned = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        
        # Surveillance du mode de vol PX4
        self._px4_flight_mode = FlightMode.UNKNOWN
        self._flight_mode_monitor_task = None
        self._offboard_expected = False  # True quand on attend le mode offboard
        self._rtl_expected = False       # True quand on attend le mode RTL
        
        # État de l'orbite thermique (loiter)
        self.is_orbiting = False
        self.orbit_mode = None  # 'evaluation' ou 'soaring'
        self.orbit_thermal_id = None
        self.orbit_thermal_obj = None  # Objet Thermal pour évaluation analytique
        self.orbit_speed = None  # Vitesse tangentielle commandée
        self.orbit_center_enu = None  # Centre de l'orbite en ENU (x, y, z)
        self.orbit_radius = None
        self.orbit_start_altitude = None  # Altitude au début de l'orbite
        self.orbit_altitude_history = []  # Historique d'altitude pendant l'orbite
        self.orbit_start_time = None
        self.altitude_monitor_task = None
        self.orbit_target_altitude_amsl = None  # Altitude AMSL cible (mise à jour pendant soaring)
        
        # Flag atterrissage : True dès que return_and_land est appelé
        self.is_landing = False
    
    async def connect(self):
        """Connexion à l'instance PX4 SITL"""
        connection_string = f"Mavsdk Server on port {self.mavsdk_port}"
        print(f"[UAV {self.uav_id}] Connexion sur {connection_string}...")
        
        self.drone = System(mavsdk_server_address="127.0.0.1", port=self.mavsdk_port)
        await self.drone.connect()
        
        # Attendre la connexion
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"[UAV {self.uav_id}] ✓ Connecté!")
                self.is_connected = True
                break
        
        # Attendre le GPS
        print(f"[UAV {self.uav_id}] Attente du GPS...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print(f"[UAV {self.uav_id}] ✓ GPS obtenu!")
                break
        
        # Attendre que le drone soit armable (tous les pré-checks passés)
        print(f"[UAV {self.uav_id}] Attente pré-checks armement...")
        timeout_start = time.time()
        async for health in self.drone.telemetry.health():
            if health.is_armable:
                print(f"[UAV {self.uav_id}] ✓ Drone armable!")
                break
            elapsed = time.time() - timeout_start
            if elapsed > 120:
                print(f"[UAV {self.uav_id}] ⚠️  Timeout armable après 120s - tentative quand même")
                print(f"[UAV {self.uav_id}]   Health: accel_cal={health.is_accelerometer_calibration_ok}, "
                      f"mag_cal={health.is_magnetometer_calibration_ok}, "
                      f"gyro_cal={health.is_gyrometer_calibration_ok}, "
                      f"local_pos={health.is_local_position_ok}, "
                      f"global_pos={health.is_global_position_ok}, "
                      f"home_pos={health.is_home_position_ok}")
                break
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print(f"[UAV {self.uav_id}]   En attente... ({int(elapsed)}s) armable={health.is_armable}")
        
        # Stocker la position home
        async for terrain_info in self.drone.telemetry.home():
            self.home_position = terrain_info
            print(f"[UAV {self.uav_id}] ✓ Position home: {terrain_info.latitude_deg:.6f}, {terrain_info.longitude_deg:.6f}")
            break
        
        # Définir l'origine de la simulation (sera potentiellement unifiée plus tard)
        self.simulation_origin = {
            'lat': self.home_position.latitude_deg,
            'lon': self.home_position.longitude_deg,
            'alt': self.home_position.absolute_altitude_m
        }
        # Home propre du drone (toujours son véritable home pour les commandes NED PX4)
        self.own_home = {
            'lat': self.home_position.latitude_deg,
            'lon': self.home_position.longitude_deg,
            'alt': self.home_position.absolute_altitude_m
        }
        
        # Démarrer la surveillance du mode de vol PX4 en arrière-plan
        self._flight_mode_monitor_task = asyncio.create_task(self._monitor_flight_mode())
    
    async def arm_and_takeoff(self):
        """Armement et décollage"""
        
        await self.drone.action.hold()
        
        print(f"[UAV {self.uav_id}] Armement...")
        max_retries = 10
        for attempt in range(max_retries):
            try:
                await self.drone.action.arm()
                print(f"[UAV {self.uav_id}] ✓ Armé!")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[UAV {self.uav_id}] Armement refusé (tentative {attempt+1}/{max_retries}): {e}")
                    # Attendre que armable revienne True
                    await asyncio.sleep(3)
                    # Re-vérifier le health
                    async for health in self.drone.telemetry.health():
                        print(f"[UAV {self.uav_id}]   armable={health.is_armable}, "
                              f"global_pos={health.is_global_position_ok}, "
                              f"home_pos={health.is_home_position_ok}")
                        break
                else:
                    print(f"[UAV {self.uav_id}] ✗ Échec armement après {max_retries} tentatives: {e}")
                    raise
        
        target_alt = self.params['working_floor']
        await self.drone.action.set_takeoff_altitude(target_alt)
        print(f"[UAV {self.uav_id}] Décollage vers {target_alt:.0f}m...")
        await self.drone.action.takeoff()
        
        async for position in self.drone.telemetry.position():
            current_alt = position.relative_altitude_m
            print(f"[UAV {self.uav_id}] Altitude actuelle: {current_alt:.1f}m / {target_alt:.1f}m", end='\r')
            if current_alt >= target_alt:
                print(f"[UAV {self.uav_id}] ✓ Altitude atteinte: {current_alt:.1f}m")
                break
    
    async def _monitor_flight_mode(self):
        """Souscription télémétrie en arrière-plan pour suivre le mode de vol PX4."""
        try:
            async for mode in self.drone.telemetry.flight_mode():
                prev = self._px4_flight_mode
                self._px4_flight_mode = mode
                if prev != mode:
                    print(f"[UAV {self.uav_id}] Mode PX4: {prev.name} → {mode.name}")
                    # Détecter perte inattendue d'offboard
                    if (self._offboard_expected
                            and prev == FlightMode.OFFBOARD
                            and mode in (FlightMode.HOLD, FlightMode.RETURN_TO_LAUNCH, FlightMode.POSCTL)):
                        print(f"[UAV {self.uav_id}] ⚠️  Perte inattendue du mode offboard → {mode.name}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[UAV {self.uav_id}] Erreur moniteur flight mode: {e}")

    async def check_and_recover_orbit(self):
        """
        Vérifie si un drone en orbite (soaring/évaluation) a été basculé par
        PX4 en RTL (failsafe timeout).  Si oui, ré-émet la commande do_orbit
        pour reprendre l'orbite.
        
        Retourne True si recovery effectué, False sinon.
        """
        if not self.is_orbiting:
            return False
        
        current_mode = self._px4_flight_mode
        if current_mode not in (FlightMode.RETURN_TO_LAUNCH, FlightMode.POSCTL):
            return False
        
        print(f"\n🚨 UAV {self.uav_id}: Mode PX4 = {current_mode.name} pendant orbite {self.orbit_mode} → Tentative recovery orbit")
        
        # Étape 1 : stabiliser en hold
        try:
            await self.drone.action.hold()
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"[UAV {self.uav_id}] Avertissement hold pendant recovery orbit: {e}")
        
        # Étape 2 : ré-émettre do_orbit avec les paramètres actuels
        if self.orbit_center_enu is None or self.orbit_radius is None:
            print(f"[UAV {self.uav_id}] ❌ Recovery orbit impossible — données d'orbite manquantes")
            return False
        
        lat_center, lon_center, _ = self._enu_to_geodetic(
            self.orbit_center_enu['X'], self.orbit_center_enu['Y'], self.orbit_center_enu['Z']
        )
        
        orbit_speed = self.orbit_speed or self.UAV_data.get('min_airspeed', 8.0)
        altitude_amsl = self.orbit_target_altitude_amsl
        if altitude_amsl is None:
            # Fallback : lire l'altitude réelle
            try:
                async for position in self.drone.telemetry.position():
                    altitude_amsl = position.absolute_altitude_m
                    break
            except Exception:
                pass
        
        if altitude_amsl is None:
            print(f"[UAV {self.uav_id}] ❌ Recovery orbit impossible — altitude inconnue")
            return False
        
        yaw_behavior = (OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER
                        if self.orbit_mode == 'evaluation'
                        else OrbitYawBehavior.HOLD_FRONT_TANGENT_TO_CIRCLE)
        
        try:
            await self.drone.action.do_orbit(
                float(-self.orbit_radius),
                float(orbit_speed),
                yaw_behavior,
                float(lat_center),
                float(lon_center),
                float(altitude_amsl)
            )
            print(f"[UAV {self.uav_id}] ✅ Recovery orbit réussi — reprise {self.orbit_mode}")
            return True
        except Exception as e:
            print(f"[UAV {self.uav_id}] ❌ Recovery orbit échoué: {e}")
            return False

    async def check_and_recover_offboard(self):
        """
        Vérifie si le drone devrait être en mode offboard mais ne l'est
        plus (ex: failsafe HOLD/RTL suite à une glitch).
        Si oui, tente de reprendre le mode offboard.
        
        Retourne True si recovery effectué, False sinon.
        """
        # Pas de recovery si on n'attend pas le mode offboard
        if not self._offboard_expected:
            return False
        
        # Pas de recovery si en orbite (géré par check_and_recover_orbit)
        if self.is_orbiting:
            return False
        
        current_mode = self._px4_flight_mode
        
        # Modes qui indiquent une perte d'offboard
        lost_modes = (FlightMode.HOLD, FlightMode.RETURN_TO_LAUNCH, FlightMode.POSCTL)
        if current_mode not in lost_modes:
            return False
        
        print(f"\n🚨 UAV {self.uav_id}: Mode PX4 = {current_mode.name} (attendu: OFFBOARD) → Tentative recovery")
        
        # Étape 1: Passer en hold pour stabiliser
        try:
            await self.drone.action.hold()
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"[UAV {self.uav_id}] Avertissement hold pendant recovery: {e}")
        
        # Étape 2: Redémarrer offboard
        success = await self.start_offboard_mode()
        if success:
            print(f"[UAV {self.uav_id}] ✅ Recovery offboard réussi")
        else:
            print(f"[UAV {self.uav_id}] ❌ Recovery offboard échoué — passage en hold")
            try:
                await self.drone.action.hold()
            except:
                pass
        return success

    async def check_and_recover_rtl(self):
        """
        Vérifie si le drone devrait être en mode RTL mais ne l'est
        plus (ex: passage en HOLD au lieu de RTL suite à un glitch).
        Si oui, tente de repasser en RTL.
        
        Même logique que check_and_recover_offboard, adaptée au RTL.
        
        Retourne True si recovery effectué, False sinon.
        """
        # Pas de recovery si on n'attend pas le mode RTL
        if not self._rtl_expected:
            return False
        
        current_mode = self._px4_flight_mode
        
        # Si déjà en RTL ou posé, pas de recovery nécessaire
        if current_mode == FlightMode.RETURN_TO_LAUNCH:
            return False
        
        # Le drone est posé (LAND ou les modes post-atterrissage)
        if current_mode in (FlightMode.LAND,):
            return False
        
        # Modes qui indiquent une perte de RTL
        lost_modes = (FlightMode.HOLD, FlightMode.POSCTL, FlightMode.OFFBOARD,
                      FlightMode.STABILIZED, FlightMode.ALTCTL)
        if current_mode not in lost_modes:
            return False
        
        print(f"\n🚨 UAV {self.uav_id}: Mode PX4 = {current_mode.name} (attendu: RTL) → Tentative recovery")
        
        # Étape 1: Passer en hold pour stabiliser
        try:
            await self.drone.action.hold()
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"[UAV {self.uav_id}] Avertissement hold pendant recovery RTL: {e}")
        
        # Étape 2: Relancer RTL
        try:
            await self.drone.action.return_to_launch()
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f"[UAV {self.uav_id}] Erreur relance RTL: {e}")
            return False
        
        # Vérifier le résultat
        if self._px4_flight_mode == FlightMode.RETURN_TO_LAUNCH:
            print(f"[UAV {self.uav_id}] ✅ Recovery RTL réussi")
            return True
        else:
            print(f"[UAV {self.uav_id}] ❌ Recovery RTL échoué (mode: {self._px4_flight_mode.name})")
            return False

    async def _offboard_keepalive_loop(self):
        """
        Boucle keepalive pour maintenir le mode offboard actif
        Envoie des setpoints à 10Hz minimum.
        IMPORTANT: N'envoyer QUE set_position_velocity_ned.
        Envoyer set_velocity_ned en plus fait basculer PX4 en mode
        velocity-only, ce qui ignore la composante position et empêche
        le drone de rejoindre sa cible.
        """
        print(f"[UAV {self.uav_id}] Démarrage keepalive offboard...")
        consecutive_errors = 0
        try:
            while True:
                try:
                    await self.drone.offboard.set_position_velocity_ned(
                        self.last_position_ned,
                        self.last_velocity_ned
                    )
                    consecutive_errors = 0
                except asyncio.CancelledError:
                    raise  # Propager pour sortir proprement
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        print(f"[UAV {self.uav_id}] ⚠️  Erreur keepalive (tentative {consecutive_errors}): {e}")
                    if consecutive_errors > 50:  # ~2.5s d'échecs à 20Hz
                        print(f"[UAV {self.uav_id}] ❌ Keepalive: trop d'erreurs consécutives ({consecutive_errors}), arrêt")
                        break
                await asyncio.sleep(0.05)  # 20Hz pour la sécurité (minimum requis: 2Hz)
        except asyncio.CancelledError:
            print(f"[UAV {self.uav_id}] Arrêt keepalive offboard")
    
    def _validate_command_values(self, north, east, down, yaw, vn, ve, vd):
        """
        Valider toutes les valeurs avant envoi à PX4
        Retourne True si valide, False sinon
        """
        # Vérifier NaN et Inf
        values = [north, east, down, yaw, vn, ve, vd]
        if any(np.isnan(v) or np.isinf(v) for v in values):
            print(f"[UAV {self.uav_id}] ⚠️  Valeurs invalides détectées (NaN/Inf):")
            print(f"   Position NED: ({north:.2f}, {east:.2f}, {down:.2f})")
            print(f"   Vitesse NED: ({vn:.2f}, {ve:.2f}, {vd:.2f})")
            print(f"   Yaw: {yaw:.2f}°")
            return False
        
        # Limites de sécurité pour les positions (m)
        MAX_POSITION = 50000.0  # 50km
        if abs(north) > MAX_POSITION or abs(east) > MAX_POSITION:
            print(f"[UAV {self.uav_id}] ⚠️  Position hors limites: N={north:.0f}m, E={east:.0f}m")
            return False
        
        # Limites d'altitude (m, down est négatif pour monter)
        MAX_ALTITUDE_DOWN = -2000.0  # 2000m max altitude
        MIN_ALTITUDE_DOWN = 5000.0   # Pas sous -5000m
        if down < MAX_ALTITUDE_DOWN or down > MIN_ALTITUDE_DOWN:
            print(f"[UAV {self.uav_id}] ⚠️  Altitude invalide: down={down:.0f}m (alt={-down:.0f}m)")
            return False
        
        # Limites de vitesse (m/s)
        MAX_VELOCITY = 50.0  # 50 m/s = 180 km/h
        if abs(vn) > MAX_VELOCITY or abs(ve) > MAX_VELOCITY or abs(vd) > MAX_VELOCITY:
            print(f"[UAV {self.uav_id}] ⚠️  Vitesse excessive: ({vn:.1f}, {ve:.1f}, {vd:.1f}) m/s")
            return False
        
        # Yaw doit être dans [-180, 180] ou [0, 360]
        if abs(yaw) > 360.0:
            print(f"[UAV {self.uav_id}] ⚠️  Yaw invalide: {yaw:.1f}°")
            return False
        
        return True
    
    async def update_from_simulation_state(self, FLT_track, FLT_conditions):
        """
        Mettre à jour PX4 avec l'état de simulation complet
        Utilise pymap3d pour les conversions de coordonnées
        """
        if len(FLT_track['X']) == 0:
            return
        
        # Position cartésienne locale (East, North, Up dans votre simulation)
        x_local = FLT_track['X'][-1]  # East
        y_local = FLT_track['Y'][-1]  # North
        z_local = FLT_track['Z'][-1]  # Up (altitude)
        
        # VALIDATION : Vérifier que les données de simulation sont valides
        if np.isnan(x_local) or np.isnan(y_local) or np.isnan(z_local):
            print(f"[UAV {self.uav_id}] ❌ Position simulation invalide: ({x_local}, {y_local}, {z_local})")
            return  # Ne pas envoyer de commande invalide à PX4
        
        if np.isinf(x_local) or np.isinf(y_local) or np.isinf(z_local):
            print(f"[UAV {self.uav_id}] ❌ Position simulation infinie: ({x_local}, {y_local}, {z_local})")
            return
        
        # Origine de référence (partagée pour coordonnées ENU cohérentes)
        lat0 = self.simulation_origin['lat']
        lon0 = self.simulation_origin['lon']
        h0 = self.simulation_origin['alt']
        
        # Convertir position locale (ENU) en géographique
        # pymap3d utilise (East, North, Up)
        lat_deg, lon_deg, alt_msl = pm.enu2geodetic(
            x_local, y_local, z_local,
            lat0, lon0, h0
        )
        
        # Convertir en ECEF pour obtenir les coordonnées absolues
        x_ecef, y_ecef, z_ecef = pm.geodetic2ecef(lat_deg, lon_deg, alt_msl)
        
        # Convertir ECEF en NED par rapport au HOME PROPRE du drone
        # PX4 offboard NED est relatif au frame local du drone (son home)
        home_lat = self.own_home['lat']
        home_lon = self.own_home['lon']
        home_alt = self.own_home['alt']
        north_ned, east_ned, down_ned = pm.ecef2ned(
            x_ecef, y_ecef, z_ecef,
            home_lat, home_lon, home_alt
        )
        
        # Orientation (bearing en radians)
        bearing_rad = FLT_track['bearing'][-1]
        
        # VALIDATION : Vérifier le bearing
        if np.isnan(bearing_rad) or np.isinf(bearing_rad):
            print(f"[UAV {self.uav_id}] ❌ Bearing invalide: {bearing_rad}")
            return
        
        # Convertir le bearing de la convention math (0=Est, CCW) vers
        # la convention navigation PX4 (0=Nord, CW) : yaw_nav = 90° - yaw_math
        yaw_deg = (90.0 - np.degrees(bearing_rad)) % 360.0
        
        # Vitesse aérodynamique
        airspeed = FLT_conditions['airspeed']
        flight_path_angle = FLT_conditions['flight_path_angle']
        
        # VALIDATION : Vérifier les vitesses
        if np.isnan(airspeed) or np.isinf(airspeed) or airspeed < 0:
            print(f"[UAV {self.uav_id}] ❌ Airspeed invalide: {airspeed}")
            return
        
        if np.isnan(flight_path_angle) or np.isinf(flight_path_angle):
            print(f"[UAV {self.uav_id}] ❌ Flight path angle invalide: {flight_path_angle}")
            return
        
        # Calculer les composantes de vitesse dans le référentiel local (ENU)
        # Vitesse verticale : utiliser le delta d'altitude réel entre les 2 dernières
        # positions FLT_track au lieu de flight_path_angle (qui vaut 0.0 en glide,
        # ce qui empêchait PX4 de descendre → drone restait à altitude constante).
        if len(FLT_track['Z']) >= 2:
            dz = FLT_track['Z'][-1] - FLT_track['Z'][-2]
            time_step = self.params.get('time_step', 1.1)
            velocity_up = dz / time_step  # m/s, négatif en descente
        else:
            velocity_up = airspeed * np.sin(flight_path_angle)
        
        # Vitesse horizontale (conserver la majorité d'airspeed pour l'horizontal)
        v_horizontal_sq = max(airspeed**2 - velocity_up**2, 0.0)
        v_horizontal = np.sqrt(v_horizontal_sq)
        
        # Composantes Est et Nord basées sur le bearing
        # Convention math de la simulation : 0=Est, X=Est=cos, Y=Nord=sin
        velocity_east = v_horizontal * np.cos(bearing_rad)
        velocity_north = v_horizontal * np.sin(bearing_rad)
        
        # Convertir les vitesses ENU en NED
        # NED: North, East, Down (Down est l'opposé de Up)
        velocity_north_ned = velocity_north
        velocity_east_ned = velocity_east
        velocity_down_ned = -velocity_up  # Down = -Up
        
        # VALIDATION FINALE : Vérifier toutes les valeurs avant envoi à PX4
        if not self._validate_command_values(
            north_ned, east_ned, down_ned, yaw_deg,
            velocity_north_ned, velocity_east_ned, velocity_down_ned
        ):
            print(f"[UAV {self.uav_id}] ⚠️  Commande rejetée - conservation dernière position valide")
            return  # Ne pas mettre à jour avec des valeurs invalides
        
        # Stocker les dernières valeurs VALIDES pour le keepalive
        # Cast float() obligatoire : pymap3d/numpy retournent np.float64
        # que gRPC/protobuf peut échouer à sérialiser → crash keepalive → RTL
        self.last_position_ned = PositionNedYaw(
            float(north_ned), float(east_ned), float(down_ned), float(yaw_deg)
        )
        self.last_velocity_ned = VelocityNedYaw(
            float(velocity_north_ned), float(velocity_east_ned),
            float(velocity_down_ned), float(yaw_deg)
        )
        
        # La commande est maintenant envoyée par la boucle keepalive
        # pas besoin d'envoyer ici pour éviter les conflits
    
    async def start_offboard_mode(self):
        """Démarrer le mode offboard avec contrôle en position+vitesse"""
        print(f"[UAV {self.uav_id}] Démarrage mode offboard ...")
        
        # Lire la position réelle actuelle du drone pour initialiser le setpoint
        # Cela évite d'envoyer (0,0,0) qui commande une descente au sol et fait
        # sortir PX4 du mode offboard par sécurité.
        try:
            async for position in self.drone.telemetry.position():
                home_lat = self.own_home['lat']
                home_lon = self.own_home['lon']
                home_alt = self.own_home['alt']
                x_ecef, y_ecef, z_ecef = pm.geodetic2ecef(
                    position.latitude_deg, position.longitude_deg,
                    position.absolute_altitude_m
                )
                north, east, down = pm.ecef2ned(
                    x_ecef, y_ecef, z_ecef,
                    home_lat, home_lon, home_alt
                )
                # Bearing initial: utiliser le yaw actuel si disponible
                yaw_deg = 0.0
                async for heading in self.drone.telemetry.heading():
                    yaw_deg = heading.heading_deg
                    break
                self.last_position_ned = PositionNedYaw(
                    float(north), float(east), float(down), float(yaw_deg)
                )
                self.last_velocity_ned = VelocityNedYaw(0.0, 0.0, 0.0, float(yaw_deg))
                print(f"[UAV {self.uav_id}] Position initiale NED: N={north:.1f}, E={east:.1f}, D={down:.1f}")
                break
        except Exception as e:
            print(f"[UAV {self.uav_id}] ⚠️  Impossible de lire la position actuelle: {e}")
        
        # Envoyer le setpoint initial avec position+vitesse
        # (même type que le keepalive pour éviter un changement de mode PX4)
        await self.drone.offboard.set_position_velocity_ned(
            self.last_position_ned, self.last_velocity_ned
        )
        
        # Démarrer la boucle keepalive AVANT offboard.start()
        # PX4 exige un flux continu de setpoints pour rester en offboard.
        # Si le keepalive démarre après, les autres tâches asyncio peuvent
        # monopoliser l'event loop et PX4 sort d'offboard par timeout.
        self.offboard_keepalive_task = asyncio.create_task(self._offboard_keepalive_loop())
        await asyncio.sleep(0.5)  # Laisser ~10 setpoints passer avant le mode switch
        
        try:
            await self.drone.offboard.start()
            self._offboard_expected = True
            print(f"[UAV {self.uav_id}] ✓ Mode offboard activé")
            return True
        except OffboardError as error:
            print(f"[UAV {self.uav_id}] ✗ Erreur offboard: {error}")
            # Annuler le keepalive si offboard a échoué
            if self.offboard_keepalive_task:
                self.offboard_keepalive_task.cancel()
                try:
                    await self.offboard_keepalive_task
                except asyncio.CancelledError:
                    pass
                self.offboard_keepalive_task = None
            return False
    
    async def stop_offboard_mode(self):
        """Arrêter le mode offboard"""
        self._offboard_expected = False
        # Arrêter la boucle keepalive
        if self.offboard_keepalive_task:
            self.offboard_keepalive_task.cancel()
            try:
                await self.offboard_keepalive_task
            except asyncio.CancelledError:
                pass
            self.offboard_keepalive_task = None
        
        try:
            await self.drone.offboard.stop()
        except:
            pass
    
    async def return_and_land(self):
        """Retour et atterrissage avec vérification RTL"""
        self.is_landing = True
        print(f"[UAV {self.uav_id}] Retour à la base...")
        # Arrêter l'orbite et le moniteur d'altitude si en cours
        if self.is_orbiting:
            if self.altitude_monitor_task:
                self.altitude_monitor_task.cancel()
                try:
                    await self.altitude_monitor_task
                except asyncio.CancelledError:
                    pass
                self.altitude_monitor_task = None
            self.is_orbiting = False
            self.orbit_mode = None
        await self.stop_offboard_mode()

        # Activer le flag RTL attendu (utilisé par check_and_recover_rtl)
        self._rtl_expected = True

        # Commande RTL avec vérification du mode effectif
        max_rtl_retries = 5
        for attempt in range(1, max_rtl_retries + 1):
            try:
                await self.drone.action.return_to_launch()
            except Exception as e:
                print(f"[UAV {self.uav_id}] ⚠️  Erreur RTL (tentative {attempt}): {e}")

            # Attendre un court instant puis vérifier le mode de vol
            await asyncio.sleep(1.0)
            current_mode = self._px4_flight_mode
            if current_mode == FlightMode.RETURN_TO_LAUNCH:
                print(f"[UAV {self.uav_id}] ✓ Mode RTL confirmé (tentative {attempt})")
                break
            else:
                print(f"[UAV {self.uav_id}] ⚠️  Mode actuel: {current_mode.name} au lieu de RTL (tentative {attempt}/{max_rtl_retries})")
                if attempt < max_rtl_retries:
                    # Passer en HOLD avant de retenter RTL (même pattern que recover offboard)
                    try:
                        await self.drone.action.hold()
                        await asyncio.sleep(0.5)
                    except Exception:
                        pass
        else:
            # Toutes les tentatives échouées
            print(f"[UAV {self.uav_id}] ❌ Impossible de passer en RTL après {max_rtl_retries} tentatives (mode: {self._px4_flight_mode.name})")

        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                print(f"[UAV {self.uav_id}] ✓ Atterri")
                break
            # Pendant l'attente atterrissage, vérifier périodiquement le mode RTL
            await self.check_and_recover_rtl()
            await asyncio.sleep(1)
        
        # Désactiver le flag RTL
        self._rtl_expected = False

    def _enu_to_geodetic(self, x_enu, y_enu, z_enu):
        """Convertir coordonnées ENU locales en lat/lon/alt géodésiques"""
        lat0 = self.simulation_origin['lat']
        lon0 = self.simulation_origin['lon']
        h0 = self.simulation_origin['alt']
        lat_deg, lon_deg, alt_msl = pm.enu2geodetic(x_enu, y_enu, z_enu, lat0, lon0, h0)
        return lat_deg, lon_deg, alt_msl

    async def start_orbit_loiter(self, thermal_center_enu, radius_m, velocity_ms, 
                                  altitude_enu, mode='evaluation', thermal_id=None,
                                  thermal_obj=None):
        """
        Démarrer un loiter en cercle autour d'un thermique via MAVSDK do_orbit.
        Arrête le mode offboard et passe en mode orbit (hold/loiter).
        
        Args:
            thermal_center_enu (dict): Centre du thermique {'X': float, 'Y': float}
            radius_m (float): Rayon du cercle en mètres (positif = horaire, négatif = anti-horaire)
            velocity_ms (float): Vitesse tangentielle en m/s
            altitude_enu (float): Altitude de l'orbite en coordonnées ENU (Up)
            mode (str): 'evaluation' ou 'soaring'
            thermal_id: Identifiant du thermique
            thermal_obj: Objet Thermal pour évaluation analytique
        """
        print(f"[UAV {self.uav_id}] Démarrage orbit/loiter ({mode}) autour du thermique {thermal_id}")
        print(f"  Centre ENU: ({thermal_center_enu['X']:.1f}, {thermal_center_enu['Y']:.1f})")
        print(f"  Rayon: {radius_m:.1f}m, Vitesse: {velocity_ms:.1f}m/s, Altitude: {altitude_enu:.1f}m")
        
        # Arrêter le mode offboard avant de passer en orbit
        await self.stop_offboard_mode()
        
        # Lire l'altitude réelle PX4 AVANT de convertir, pour l'utiliser comme altitude d'orbite
        current_abs_altitude = None
        async for position in self.drone.telemetry.position():
            self.orbit_start_altitude = position.relative_altitude_m
            current_abs_altitude = position.absolute_altitude_m
            print(f"[UAV {self.uav_id}] Altitude de départ orbite: {self.orbit_start_altitude:.1f}m (abs: {current_abs_altitude:.1f}m)")
            break
        
        # Convertir le centre du thermique de ENU en coordonnées géodésiques
        # On utilise altitude_enu juste pour le centre, mais l'altitude d'orbite sera l'altitude réelle
        lat_center, lon_center, _alt_enu_converted = self._enu_to_geodetic(
            thermal_center_enu['X'], thermal_center_enu['Y'], altitude_enu
        )
        
        # Utiliser l'altitude absolue réelle du drone (pas l'altitude ENU de simulation)
        # pour éviter que l'orbite force le drone à monter ou descendre
        if current_abs_altitude is not None:
            alt_center = current_abs_altitude
        else:
            alt_center = _alt_enu_converted
            print(f"[UAV {self.uav_id}] ⚠️  Altitude réelle non disponible, utilisation altitude ENU convertie")
        
        print(f"[UAV {self.uav_id}] Altitude orbite AMSL: {alt_center:.1f}m")
        
        # Stocker l'état de l'orbite
        self.is_orbiting = True
        self.orbit_mode = mode
        self.orbit_thermal_id = thermal_id
        self.orbit_thermal_obj = thermal_obj
        self.orbit_speed = velocity_ms
        self.orbit_center_enu = {'X': thermal_center_enu['X'], 'Y': thermal_center_enu['Y'], 'Z': altitude_enu}
        self.orbit_radius = abs(radius_m)
        self.orbit_altitude_history = []
        self.orbit_start_time = time.time()
        self.orbit_target_altitude_amsl = alt_center
        
        # Lancer l'orbite avec do_orbit
        # Yaw: face au centre pendant l'évaluation, tangentiel pendant le soaring
        if mode == 'evaluation':
            yaw_behavior = OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER
        else:
            yaw_behavior = OrbitYawBehavior.HOLD_FRONT_TANGENT_TO_CIRCLE

        # Convertir tous les paramètres en float natif Python
        # pymap3d retourne des np.float64 que gRPC/protobuf ne sérialise pas
        radius_m_f = float(radius_m)
        velocity_ms_f = float(velocity_ms)
        lat_center_f = float(lat_center)
        lon_center_f = float(lon_center)
        alt_center_f = float(alt_center)

        try:
            await self.drone.action.do_orbit(
                radius_m_f,        # Rayon (négatif = sens anti-horaire)
                velocity_ms_f,     # Vitesse tangentielle
                yaw_behavior,      # Comportement du yaw
                lat_center_f,      # Latitude du centre
                lon_center_f,      # Longitude du centre
                alt_center_f       # Altitude absolue (AMSL)
            )
            print(f"[UAV {self.uav_id}] ✓ Orbit/loiter démarré ({mode})")
        except Exception as e:
            print(f"[UAV {self.uav_id}] ✗ Erreur do_orbit: {e} → Reprise offboard")
            self.is_orbiting = False
            self.orbit_mode = None
            self.orbit_thermal_id = None
            # Relancer offboard puisqu'il a été arrêté avant do_orbit
            try:
                await self.start_offboard_mode()
            except Exception as e2:
                print(f"[UAV {self.uav_id}] ✗ Échec reprise offboard après do_orbit: {e2}")
                # Dernier recours : tenter hold
                try:
                    await self.drone.action.hold()
                except Exception:
                    pass
            return False
        
        # Démarrer la surveillance d'altitude
        self.altitude_monitor_task = asyncio.create_task(self._altitude_monitor_loop())
        
        return True

    async def _altitude_monitor_loop(self):
        """
        Boucle de surveillance d'altitude pendant l'orbite.
        Enregistre l'altitude à intervalles réguliers pour évaluer la thermique.
        """
        print(f"[UAV {self.uav_id}] Démarrage surveillance altitude...")
        try:
            async for position in self.drone.telemetry.position():
                if not self.is_orbiting:
                    break
                
                current_alt = position.relative_altitude_m
                elapsed = time.time() - self.orbit_start_time
                
                self.orbit_altitude_history.append({
                    'time': elapsed,
                    'altitude': current_alt,
                    'timestamp': time.time()
                })
                
                # Log toutes les 5 secondes
                if len(self.orbit_altitude_history) % 50 == 0:
                    alt_change = current_alt - self.orbit_start_altitude
                    print(f"[UAV {self.uav_id}] Orbite {self.orbit_mode}: alt={current_alt:.1f}m, "
                          f"Δalt={alt_change:+.1f}m, durée={elapsed:.1f}s")
                
                await asyncio.sleep(0.1)  # 10Hz sampling
                
        except asyncio.CancelledError:
            print(f"[UAV {self.uav_id}] Arrêt surveillance altitude")
        except Exception as e:
            print(f"[UAV {self.uav_id}] Erreur surveillance altitude: {e}")

    def get_altitude_variation(self):
        """
        Calculer la variation d'altitude pendant l'orbite.
        
        Returns:
            dict: {
                'total_change': float,     # Changement total d'altitude (m)
                'avg_climb_rate': float,   # Taux de montée moyen (m/s)
                'max_altitude': float,     # Altitude maximale atteinte
                'min_altitude': float,     # Altitude minimale
                'duration': float,         # Durée de l'orbite (s)
                'is_climbing': bool,       # True si le drone monte
                'samples': int             # Nombre d'échantillons
            }
        """
        if not self.orbit_altitude_history or len(self.orbit_altitude_history) < 2:
            return {
                'total_change': 0.0,
                'avg_climb_rate': 0.0,
                'max_altitude': self.orbit_start_altitude or 0.0,
                'min_altitude': self.orbit_start_altitude or 0.0,
                'duration': 0.0,
                'is_climbing': False,
                'samples': 0
            }
        
        altitudes = [h['altitude'] for h in self.orbit_altitude_history]
        times = [h['time'] for h in self.orbit_altitude_history]
        
        # Utiliser orbit_start_altitude comme référence (pas le premier échantillon)
        # car le monitoring démarre avec un délai après le début effectif de l'orbite
        ref_altitude = self.orbit_start_altitude if self.orbit_start_altitude is not None else altitudes[0]
        total_change = altitudes[-1] - ref_altitude
        # Durée totale depuis le début de l'orbite (times[i] = elapsed depuis orbit_start_time)
        duration = times[-1] if times[-1] > 0 else (times[-1] - times[0])
        avg_climb_rate = total_change / duration if duration > 0 else 0.0
        
        return {
            'total_change': total_change,
            'avg_climb_rate': avg_climb_rate,
            'max_altitude': max(altitudes),
            'min_altitude': min(altitudes),
            'duration': duration,
            'is_climbing': total_change > 0.5,  # Seuil de 0.5m pour éviter le bruit
            'samples': len(altitudes)
        }

    async def evaluate_thermal_from_orbit(self, min_evaluation_time=15.0, min_climb_threshold=0.3):
        """
        Évaluer si le thermique est exploitable.
        
        IMPORTANT: PX4 do_orbit maintient une altitude fixe, donc les thermiques simulées
        ne produisent aucun changement d'altitude mesurable via télémétrie.
        L'évaluation utilise le modèle analytique du thermique (lift_rate vs sink_rate).
        
        Args:
            min_evaluation_time (float): Temps minimum d'évaluation en secondes
            min_climb_threshold (float): Taux de montée minimum (m/s) pour valider le thermique
            
        Returns:
            dict: {
                'evaluation_complete': bool,
                'thermal_viable': bool,
                'altitude_gain': float,
                'climb_rate': float,
                'evaluation_time': float
            }
        """
        if not self.is_orbiting or self.orbit_mode != 'evaluation':
            return {'evaluation_complete': False, 'thermal_viable': False, 
                    'altitude_gain': 0.0, 'climb_rate': 0.0, 'evaluation_time': 0.0}
        
        elapsed = time.time() - self.orbit_start_time
        
        # Pas assez de données encore
        if elapsed < min_evaluation_time:
            return {'evaluation_complete': False, 'thermal_viable': False,
                    'altitude_gain': 0.0, 'climb_rate': 0.0, 'evaluation_time': elapsed}
        
        # --- Évaluation analytique via le modèle de thermique ---
        # PX4 do_orbit maintient l'altitude → impossible de mesurer le lift via télémétrie.
        # On calcule le taux de montée net = portance_thermique - taux_de_chute_planeur.
        if self.orbit_thermal_obj is not None and self.orbit_radius is not None:
            orbit_radius = self.orbit_radius
            thermal = self.orbit_thermal_obj
            orbit_speed = self.orbit_speed or self.UAV_data.get('min_airspeed', 12.0)
            
            # Portance du thermique au rayon d'orbite
            lift_rate = thermal.get_lift_rate(orbit_radius)
            
            # Conditions atmosphériques
            working_floor = self.params.get('working_floor', 600.0)
            atm = _compute_atmosphere(working_floor)
            grav = atm['grav']
            air_density = atm['air_density']
            
            # Angle d'inclinaison pour orbite circulaire: tan(bank) = v²/(g×r)
            bank_angle = np.arctan2(orbit_speed**2, grav * orbit_radius)
            
            flight_conditions_orbit = {
                'airspeed': orbit_speed,
                'weight': 0.0,
                'flight_path_angle': 0.0,
                'grav_accel': grav,
                'bank_angle': bank_angle,
                'airspeed_dot': 0.0,
                'air_density': air_density
            }
            sink_rate = abs(get_sink_rate(self.UAV_data, flight_conditions_orbit))
            
            net_climb_rate = lift_rate - sink_rate
            estimated_alt_gain = net_climb_rate * elapsed
            
            print(f"[UAV {self.uav_id}] Évaluation thermique {self.orbit_thermal_id} (analytique):")
            print(f"  Rayon orbite: {orbit_radius:.1f}m | Portance: {lift_rate:.2f}m/s | Chute: {sink_rate:.2f}m/s")
            print(f"  Angle inclinaison: {np.degrees(bank_angle):.1f}° | Vitesse: {orbit_speed:.1f}m/s")
            print(f"  Taux montée net: {net_climb_rate:+.2f}m/s | Gain estimé: {estimated_alt_gain:+.1f}m en {elapsed:.1f}s")
            print(f"  Thermique {'VIABLE ✓' if net_climb_rate >= min_climb_threshold else 'NON VIABLE ✗'}")
            
            return {
                'evaluation_complete': True,
                'thermal_viable': net_climb_rate >= min_climb_threshold,
                'altitude_gain': estimated_alt_gain,
                'climb_rate': net_climb_rate,
                'evaluation_time': elapsed
            }
        
        # Fallback: si pas d'objet thermique, utiliser la télémétrie (ancien comportement)
        alt_var = self.get_altitude_variation()
        
        result = {
            'evaluation_complete': True,
            'thermal_viable': alt_var['avg_climb_rate'] >= min_climb_threshold,
            'altitude_gain': alt_var['total_change'],
            'climb_rate': alt_var['avg_climb_rate'],
            'evaluation_time': elapsed
        }
        
        print(f"[UAV {self.uav_id}] Évaluation thermique {self.orbit_thermal_id} (télémétrie):")
        print(f"  Gain d'altitude: {alt_var['total_change']:+.1f}m en {elapsed:.1f}s")
        print(f"  Taux de montée moyen: {alt_var['avg_climb_rate']:.2f}m/s")
        print(f"  Thermique {'VIABLE ✓' if result['thermal_viable'] else 'NON VIABLE ✗'}")
        
        return result

    async def transition_to_soaring_orbit(self, optimal_radius, optimal_speed):
        """
        Transition de l'orbite d'évaluation vers l'orbite de soaring
        avec les paramètres optimaux calculés.
        
        Args:
            optimal_radius (float): Rayon optimal d'orbite (m)
            optimal_speed (float): Vitesse optimale (m/s)
        """
        if not self.is_orbiting or not self.orbit_center_enu:
            print(f"[UAV {self.uav_id}] Impossible de transitionner - pas en orbite")
            return False
        
        print(f"[UAV {self.uav_id}] Transition évaluation → soaring")
        print(f"  Nouveau rayon: {optimal_radius:.1f}m, Vitesse: {optimal_speed:.1f}m/s")
        
        # Convertir le centre en géodésique
        lat_center, lon_center, _alt_center = self._enu_to_geodetic(
            self.orbit_center_enu['X'], self.orbit_center_enu['Y'], self.orbit_center_enu['Z']
        )
        
        # Utiliser l'altitude réelle actuelle pour ne pas forcer un changement d'altitude
        current_abs_altitude = _alt_center
        try:
            async for position in self.drone.telemetry.position():
                current_abs_altitude = position.absolute_altitude_m
                break
        except Exception:
            pass
        
        self.orbit_mode = 'soaring'
        self.orbit_radius = optimal_radius
        self.orbit_target_altitude_amsl = current_abs_altitude
        
        # Relancer l'orbite avec les paramètres optimaux
        # Convertir en float natif Python pour gRPC/protobuf
        try:
            await self.drone.action.do_orbit(
                float(-optimal_radius),     # Négatif = sens anti-horaire (standard thermique)
                float(optimal_speed),
                OrbitYawBehavior.HOLD_FRONT_TANGENT_TO_CIRCLE,
                float(lat_center),
                float(lon_center),
                float(current_abs_altitude)
            )
            print(f"[UAV {self.uav_id}] ✓ Transition soaring réussie")
            return True
        except Exception as e:
            print(f"[UAV {self.uav_id}] ✗ Erreur transition soaring: {e}")
            return False

    async def update_soaring_altitude(self, time_step, z_upper_bound=1000.0):
        """
        Met à jour l'altitude de l'orbite soaring progressivement
        pour simuler la montée dans le thermique.
        PX4 do_orbit maintient l'altitude → on doit la ré-émettre périodiquement.
        
        Args:
            time_step (float): Pas de temps simulation (secondes)
            z_upper_bound (float): Altitude max ENU (m)
        """
        if not self.is_orbiting or self.orbit_mode != 'soaring':
            return
        if self.orbit_thermal_obj is None or self.orbit_radius is None:
            return
        if self.orbit_target_altitude_amsl is None:
            return
        
        thermal = self.orbit_thermal_obj
        orbit_radius = self.orbit_radius
        orbit_speed = self.orbit_speed or self.UAV_data.get('min_airspeed', 8.0)
        
        # Portance du thermique au rayon d'orbite
        lift_rate = thermal.get_lift_rate(orbit_radius)
        
        # Conditions atmosphériques et taux de chute
        working_floor = self.params.get('working_floor', 600.0)
        atm = _compute_atmosphere(working_floor)
        grav = atm['grav']
        air_density = atm['air_density']
        
        bank_angle = np.arctan2(orbit_speed**2, grav * orbit_radius)
        flight_cond = {
            'airspeed': orbit_speed,
            'weight': 0.0,
            'flight_path_angle': 0.0,
            'grav_accel': grav,
            'bank_angle': bank_angle,
            'airspeed_dot': 0.0,
            'air_density': air_density
        }
        sink_rate = abs(get_sink_rate(self.UAV_data, flight_cond))
        
        net_climb = lift_rate - sink_rate
        if net_climb <= 0:
            return
        
        # Vérifier altitude max (convertir z_upper_bound ENU → approximation AMSL)
        sim_origin_alt = self.simulation_origin['alt'] if self.simulation_origin else 0.0
        max_altitude_amsl = sim_origin_alt + z_upper_bound
        if self.orbit_target_altitude_amsl >= max_altitude_amsl:
            return
        
        # Lire l'altitude réelle du drone et commander : altitude_réelle + climb × dt
        # Ainsi la cible est toujours EN AVANCE sur la position actuelle,
        # ce qui évite que le drone stagne à une altitude intermédiaire.
        current_abs_altitude = self.orbit_target_altitude_amsl
        try:
            async for position in self.drone.telemetry.position():
                current_abs_altitude = position.absolute_altitude_m
                break
        except Exception:
            pass
        
        altitude_gain = net_climb * time_step
        new_altitude = min(current_abs_altitude + altitude_gain, max_altitude_amsl)
        self.orbit_target_altitude_amsl = new_altitude
        
        # Ré-émettre do_orbit avec la nouvelle altitude
        lat_center, lon_center, _ = self._enu_to_geodetic(
            self.orbit_center_enu['X'], self.orbit_center_enu['Y'], self.orbit_center_enu['Z']
        )
        
        try:
            await self.drone.action.do_orbit(
                float(-self.orbit_radius),
                float(orbit_speed),
                OrbitYawBehavior.HOLD_FRONT_TANGENT_TO_CIRCLE,
                float(lat_center),
                float(lon_center),
                float(new_altitude)
            )
        except Exception as e:
            print(f"[UAV {self.uav_id}] Erreur mise à jour altitude soaring: {e}")

    async def stop_orbit_loiter(self):
        """
        Arrêter l'orbite/loiter et reprendre le mode offboard.
        """
        if not self.is_orbiting:
            return
        
        print(f"[UAV {self.uav_id}] Arrêt orbit/loiter ({self.orbit_mode})")
        
        # Arrêter la surveillance d'altitude
        if self.altitude_monitor_task:
            self.altitude_monitor_task.cancel()
            try:
                await self.altitude_monitor_task
            except asyncio.CancelledError:
                pass
            self.altitude_monitor_task = None
        
        # Log final de variation d'altitude
        if self.orbit_altitude_history:
            alt_var = self.get_altitude_variation()
            print(f"[UAV {self.uav_id}] Résumé orbite: Δalt={alt_var['total_change']:+.1f}m, "
                  f"taux moyen={alt_var['avg_climb_rate']:.2f}m/s, durée={alt_var['duration']:.1f}s")
        
        # Réinitialiser l'état
        self.is_orbiting = False
        self.orbit_mode = None
        self.orbit_thermal_id = None
        self.orbit_thermal_obj = None
        self.orbit_speed = None
        self.orbit_center_enu = None
        self.orbit_radius = None
        self.orbit_altitude_history = []
        self.orbit_target_altitude_amsl = None
        
        # Mettre à jour last_position_ned avec la position réelle actuelle
        # pour éviter un saut brusque quand offboard reprend
        try:
            async for position in self.drone.telemetry.position():
                # NED doit être relatif au home propre du drone (pour PX4 offboard)
                home_lat = self.own_home['lat']
                home_lon = self.own_home['lon']
                home_alt = self.own_home['alt']
                x_ecef, y_ecef, z_ecef = pm.geodetic2ecef(
                    position.latitude_deg, position.longitude_deg,
                    position.absolute_altitude_m
                )
                north, east, down = pm.ecef2ned(x_ecef, y_ecef, z_ecef, home_lat, home_lon, home_alt)
                self.last_position_ned = PositionNedYaw(
                    float(north), float(east), float(down), 0.0
                )
                self.last_velocity_ned = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                print(f"[UAV {self.uav_id}] Position NED mise à jour: N={north:.1f}, E={east:.1f}, D={down:.1f}")
                break
        except Exception as e:
            print(f"[UAV {self.uav_id}] Avertissement lecture position: {e}")
        
        # Mettre en hold avant de reprendre offboard
        try:
            await self.drone.action.hold()
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"[UAV {self.uav_id}] Avertissement hold: {e}")
        
        # Reprendre le mode offboard
        success = await self.start_offboard_mode()
        if success:
            print(f"[UAV {self.uav_id}] ✓ Reprise mode offboard après orbite")
        else:
            print(f"[UAV {self.uav_id}] ⚠️  Échec reprise offboard - tentative hold")
            try:
                await self.drone.action.hold()
            except:
                pass
        
        return success

    async def get_current_position_enu(self):
        """
        Lire la position actuelle du drone via MAVSDK télémétrie et
        la convertir en coordonnées ENU locales.
        
        Returns:
            dict: {'X': float, 'Y': float, 'Z': float} en coordonnées ENU
        """
        async for position in self.drone.telemetry.position():
            lat = position.latitude_deg
            lon = position.longitude_deg
            alt = position.absolute_altitude_m
            
            lat0 = self.simulation_origin['lat']
            lon0 = self.simulation_origin['lon']
            h0 = self.simulation_origin['alt']
            
            x_enu, y_enu, z_enu = pm.geodetic2enu(lat, lon, alt, lat0, lon0, h0)
            return {'X': x_enu, 'Y': y_enu, 'Z': z_enu}

    async def get_current_telemetry(self):
        """
        Lire la position ET la vitesse actuelles du drone via MAVSDK.
        Retourne position ENU + vitesse ENU (groundspeed et vertical).
        
        Returns:
            dict: {
                'X': float, 'Y': float, 'Z': float,      # position ENU (m)
                'vn': float, 've': float, 'vd': float,    # vitesse NED (m/s)
                'groundspeed': float,                       # vitesse horizontale (m/s)
                'bearing': float,                           # cap basé sur vitesse (rad, nav convention)
                'climb_rate': float                         # taux de montée (m/s, positif = montée)
            } ou None en cas d'erreur
        """
        try:
            # Lire position
            pos_data = None
            async for position in self.drone.telemetry.position():
                lat = position.latitude_deg
                lon = position.longitude_deg
                alt = position.absolute_altitude_m
                
                lat0 = self.simulation_origin['lat']
                lon0 = self.simulation_origin['lon']
                h0 = self.simulation_origin['alt']
                
                x_enu, y_enu, z_enu = pm.geodetic2enu(lat, lon, alt, lat0, lon0, h0)
                pos_data = {'X': x_enu, 'Y': y_enu, 'Z': z_enu}
                break
            
            if pos_data is None:
                return None
            
            # Lire vitesse NED
            vel_data = {'vn': 0.0, 've': 0.0, 'vd': 0.0}
            async for velocity in self.drone.telemetry.velocity_ned():
                vel_data['vn'] = velocity.north_m_s
                vel_data['ve'] = velocity.east_m_s
                vel_data['vd'] = velocity.down_m_s
                break
            
            # Calculer les métriques dérivées
            vn = vel_data['vn']
            ve = vel_data['ve']
            vd = vel_data['vd']
            groundspeed = np.sqrt(vn**2 + ve**2)
            
            # Bearing en convention math (0=Est, CCW) cohérent avec la simulation
            # X=East, Y=North → atan2(North, East) = atan2(vn, ve)
            if groundspeed > 0.5:  # Seuil minimum pour un bearing fiable
                bearing = np.arctan2(vn, ve)
            else:
                bearing = None  # Pas de bearing fiable si quasi-stationnaire
            
            # Taux de montée (positif = montée, négatif = descente)
            climb_rate = -vd  # NED: down positif = descente
            
            return {
                **pos_data,
                **vel_data,
                'groundspeed': groundspeed,
                'bearing': bearing,
                'climb_rate': climb_rate
            }
        except Exception as e:
            print(f"[UAV {self.uav_id}] ⚠️  Erreur lecture télémétrie: {e}")
            return None


class MultiUAVController:
    """
    Contrôleur pour gérer plusieurs UAVs simultanément
    """
    
    def __init__(self, nUAVs, params, UAV_data):
        self.nUAVs = nUAVs
        self.params = params
        self.UAV_data = UAV_data
        self.bridges = []
        
        # Ports MAVLink pour chaque drone
        # Port de base: 14540, puis 14541, 14542, etc.
        self.base_port = 14540
    
    def get_connection_port(self, uav_id):
        """Obtenir le port de connexion pour un UAV"""
        return self.base_port + uav_id
    
    def get_mavsdk_port(self, uav_id):
        """Obtenir le port MAVSDK pour un UAV"""
        return 50051 + uav_id
    
    async def initialize_all_uavs(self):
        """Initialiser tous les UAVs en parallèle"""
        print("\n" + "="*70)
        print(f"INITIALISATION DE {self.nUAVs} UAVs")
        print("="*70)
        
        # Créer les bridges pour tous les UAVs
        for u in range(self.nUAVs):
            port = self.get_connection_port(u)
            mavsdk_port = self.get_mavsdk_port(u)
            bridge = PX4SITLBridge(u, self.params, self.UAV_data, port, mavsdk_port)
            self.bridges.append(bridge)
        
        # Connecter tous les UAVs en parallèle
        print("\nConnexion aux UAVs...")
        tasks = []
        for u in range(self.nUAVs):
            tasks.append(asyncio.create_task(self.bridges[u].connect()))
        await asyncio.gather(*tasks)
        
        # Unifier l'origine de simulation : tous les bridges utilisent l'origine du bridge 0
        # pour que les coordonnées ENU soient cohérentes dans toute la simulation
        if self.bridges:
            shared_origin = self.bridges[0].simulation_origin
            for u in range(1, self.nUAVs):
                self.bridges[u].simulation_origin = shared_origin
                print(f"[UAV {u}] Origine unifiée → bridge 0: "
                      f"({shared_origin['lat']:.6f}, {shared_origin['lon']:.6f}, {shared_origin['alt']:.1f})")
        
        print("\n✓ Tous les UAVs sont connectés!")
    
    async def arm_and_takeoff_all(self):
        """Faire décoller tous les UAVs"""
        print("\n" + "="*70)
        print("DÉCOLLAGE DE TOUS LES UAVs")
        print("="*70)
        
        # Armer et décoller en parallèle
        tasks = []
        for u in range(self.nUAVs):
            tasks.append(asyncio.create_task(self.bridges[u].arm_and_takeoff()))
        await asyncio.gather(*tasks)
        
        print("\n✓ Tous les UAVs sont en l'air!")
    
    async def start_offboard_all(self):
        """Démarrer le mode offboard pour tous les UAVs séquentiellement.
        
        Le démarrage séquentiel est nécessaire car:
        1. Chaque drone doit avoir son keepalive stable AVANT le mode switch
        2. En parallèle, les lectures télémétrie gRPC monopolisent l'event loop
           et affament les keepalive des drones déjà démarrés → timeout PX4
        3. Un délai entre chaque drone laisse le keepalive s'établir
        """
        print("\nDémarrage du mode offboard pour tous les UAVs (séquentiel)...")
        success_count = 0
        for u in range(self.nUAVs):
            result = await self.bridges[u].start_offboard_mode()
            if result:
                success_count += 1
            else:
                print(f"⚠️  UAV {u}: échec offboard, tentative de retry...")
                await asyncio.sleep(1.0)
                result = await self.bridges[u].start_offboard_mode()
                if result:
                    success_count += 1
            # Délai entre chaque drone pour laisser les keepalive se stabiliser
            await asyncio.sleep(0.5)
        
        if success_count == self.nUAVs:
            print(f"✓ Mode offboard activé pour tous les {self.nUAVs} UAVs")
            return True
        else:
            print(f"⚠️  Mode offboard activé pour {success_count}/{self.nUAVs} UAVs")
            return False

    async def recover_offboard_all(self, FLT_track):
        """Vérifier et récupérer le mode offboard / orbit pour tous les UAVs.
        
        - Drones en orbite : vérifier RTL failsafe → ré-émettre do_orbit
        - Drones en offboard : vérifier perte offboard → reprendre offboard
        """
        for u in range(self.nUAVs):
            if u >= len(self.bridges):
                continue
            bridge = self.bridges[u]
            if bridge.is_landing:
                continue
            
            # Drones en orbite (soaring ou évaluation) : vérifier RTL failsafe
            if bridge.is_orbiting:
                await bridge.check_and_recover_orbit()
                continue
            
            # Exclure les drones en évaluation ou soaring gérés autrement
            if FLT_track[u].get('in_evaluation', False):
                continue
            flight_mode = FLT_track[u]['flight_mode'][-1] if FLT_track[u]['flight_mode'] else 'glide'
            if flight_mode == 'soaring':
                continue
            # Tenter le recovery offboard (la méthode vérifie si c'est nécessaire)
            await bridge.check_and_recover_offboard()
    
    async def sync_all_positions_from_mavsdk(self, FLT_track, FLT_conditions):
        """
        Synchroniser position MAVSDK, vitesse (airspeed) et cap réels de
        tous les UAVs vers FLT_track et FLT_conditions.
        
        Appelée AVANT gotoWaypointMulti pour que l'algorithme de décision
        travaille avec la position réelle du drone, pas la position calculée.
        
        Synchronise :
        - Position ENU (FLT_track X, Y, Z)
        - Bearing basé sur la vitesse réelle (FLT_track bearing)
        - Airspeed / groundspeed réel (FLT_conditions airspeed)
        
        Pour les drones en orbite, ne pas modifier (sync_orbit_position_to_simulation s'en charge).
        """
        async def _sync_one_uav(u):
            bridge = self.bridges[u]
            # Ignorer les drones en atterrissage/atterris
            if bridge.is_landing:
                return
            # Les drones en orbite sont gérés séparément par sync_orbit_position_to_simulation
            if bridge.is_orbiting:
                return
            # Les drones en évaluation sont aussi en orbite
            if FLT_track[u].get('in_evaluation', False):
                return
            # Besoin d'au moins un point dans FLT_track pour écraser
            if not FLT_track[u]['X']:
                return
            
            try:
                telemetry = await bridge.get_current_telemetry()
                if telemetry is None:
                    return
                
                # Valider les coordonnées
                if any(np.isnan(v) or np.isinf(v) for v in [telemetry['X'], telemetry['Y'], telemetry['Z']]):
                    return
                
                # Écraser la dernière position dans FLT_track avec la position réelle
                FLT_track[u]['X'][-1] = telemetry['X']
                FLT_track[u]['Y'][-1] = telemetry['Y']
                FLT_track[u]['Z'][-1] = telemetry['Z']
                
                # Bearing basé sur la direction de vol PX4 (plus fiable que le delta de position)
                if telemetry['bearing'] is not None:
                    FLT_track[u]['bearing'][-1] = telemetry['bearing']
                elif len(FLT_track[u]['X']) >= 2:
                    # Fallback : delta de position si vitesse trop faible
                    # Convention math : atan2(dY, dX) = atan2(dNorth, dEast)
                    dx = FLT_track[u]['X'][-1] - FLT_track[u]['X'][-2]  # delta East
                    dy = FLT_track[u]['Y'][-1] - FLT_track[u]['Y'][-2]  # delta North
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        FLT_track[u]['bearing'][-1] = np.arctan2(dy, dx)
                
                # Synchroniser la vitesse réelle vers FLT_conditions
                # pour que gotoWaypoint utilise l'airspeed réelle de PX4
                groundspeed = telemetry['groundspeed']
                min_v = bridge.UAV_data.get('min_airspeed', 8.0)
                max_v = bridge.UAV_data.get('max_airspeed', 30.0)
                if groundspeed > 0.5:  # Seuil minimum de fiabilité
                    # Borner dans la plage admissible de l'UAV
                    clamped_speed = np.clip(groundspeed, min_v, max_v)
                    FLT_conditions[u]['airspeed'] = clamped_speed
                
            except Exception as e:
                print(f"[UAV {u}] ⚠️  Sync MAVSDK→simulation échouée: {e}")
        
        tasks = []
        for u in range(self.nUAVs):
            if u < len(self.bridges):
                tasks.append(asyncio.create_task(_sync_one_uav(u)))
        
        if tasks:
            await asyncio.gather(*tasks)

    async def update_all_from_simulation(self, FLT_track, FLT_conditions):
        """Mettre à jour tous les UAVs avec l'état de simulation
        Les UAVs en orbite (loiter) ou en évaluation sont exclus car PX4 gère leur position"""
        update_tasks = []
        for u in range(self.nUAVs):
            if u < len(self.bridges):
                # Ignorer les drones en atterrissage/atterris
                if self.bridges[u].is_landing:
                    continue
                # Ne pas envoyer de commandes offboard aux UAVs en orbite
                if self.bridges[u].is_orbiting:
                    continue
                # Ne pas envoyer de commandes offboard aux UAVs en évaluation
                # (cohérent avec sync_all_positions_from_mavsdk qui les exclut aussi)
                if FLT_track[u].get('in_evaluation', False):
                    continue
                update_tasks.append(
                    asyncio.create_task(self.bridges[u].update_from_simulation_state(FLT_track[u], FLT_conditions[u]))
                )
        
        if update_tasks:
            await asyncio.gather(*update_tasks)
    
    async def recover_rtl_all(self):
        """Vérifier et récupérer le mode RTL pour tous les UAVs qui l'ont perdu.
        
        Même logique que recover_offboard_all, adaptée au RTL.
        """
        for u in range(self.nUAVs):
            if u >= len(self.bridges):
                continue
            await self.bridges[u].check_and_recover_rtl()

    async def land_all(self):
        """Atterrir tous les UAVs avec vérification RTL"""
        print("\n" + "="*70)
        print("ATTERRISSAGE DE TOUS LES UAVs")
        print("="*70)
        tasks = []
        for u in range(self.nUAVs):
            tasks.append(asyncio.create_task(self.bridges[u].return_and_land()))
        await asyncio.gather(*tasks)
        
        # Vérification finale : s'assurer que tous les UAVs ont bien atterri
        # Utilise la même logique que recover_offboard_all : vérification par bridge
        print("\nVérification finale du statut de tous les UAVs...")
        all_landed = True
        for u in range(self.nUAVs):
            bridge = self.bridges[u]
            mode = bridge._px4_flight_mode
            try:
                async for in_air in bridge.drone.telemetry.in_air():
                    if in_air:
                        all_landed = False
                        print(f"[UAV {u}] ⚠️  Toujours en l'air (mode: {mode.name}) → Tentative recovery RTL")
                        # Utiliser check_and_recover_rtl (même pattern que offboard)
                        bridge._rtl_expected = True
                        recovered = await bridge.check_and_recover_rtl()
                        if not recovered:
                            # Fallback: forcer RTL directement
                            print(f"[UAV {u}] ⚠️  Recovery échoué, RTL direct...")
                            try:
                                await bridge.drone.action.hold()
                                await asyncio.sleep(0.3)
                                await bridge.drone.action.return_to_launch()
                            except Exception as e:
                                print(f"[UAV {u}] ❌ Échec RTL direct: {e}")
                    else:
                        print(f"[UAV {u}] ✓ Au sol")
                    break
            except Exception as e:
                print(f"[UAV {u}] ⚠️  Impossible de vérifier le statut: {e}")
        
        if not all_landed:
            print("\n⏳ Attente atterrissage des UAVs restants...")
            for u in range(self.nUAVs):
                bridge = self.bridges[u]
                try:
                    async for in_air in bridge.drone.telemetry.in_air():
                        if not in_air:
                            print(f"[UAV {u}] ✓ Atterri (vérification finale)")
                            break
                        # Vérifier périodiquement le mode RTL pendant l'attente
                        await bridge.check_and_recover_rtl()
                        await asyncio.sleep(1)
                except Exception:
                    pass
        
        # Désactiver tous les flags RTL
        for u in range(self.nUAVs):
            self.bridges[u]._rtl_expected = False
        
        print("\n✓ Tous les UAVs ont atterri!")
        
    async def set_altitude_all(self, target_altitude):
        """Définir l'altitude cible pour tous les UAVs"""
        print(f"\nDéfinition de l'altitude cible à {target_altitude}m pour tous les UAVs...")
        tasks = []
        for u in range(self.nUAVs):
            tasks.append(asyncio.create_task(self.bridges[u].drone.action.do_orbit(
                100, 15, OrbitYawBehavior.HOLD_FRONT_TANGENT_TO_CIRCLE,
                self.bridges[u].home_position.latitude_deg, 
                self.bridges[u].home_position.longitude_deg, target_altitude
            )))
        await asyncio.gather(*tasks)
        print(f"\n✓ Altitude cible définie à {target_altitude}m pour tous les UAVs!")

    async def start_thermal_orbit(self, uav_id, thermal, altitude, mode='evaluation'):
        """
        Démarrer une orbite autour d'un thermique pour un UAV spécifique.
        
        Args:
            uav_id (int): Index de l'UAV
            thermal: Objet Thermal avec x, y, radius, strength
            altitude (float): Altitude de l'orbite en ENU
            mode (str): 'evaluation' ou 'soaring'
        Returns:
            bool: True si l'orbite a démarré avec succès
        """
        if uav_id >= len(self.bridges):
            return False
        
        bridge = self.bridges[uav_id]
        
        # Centre du thermique
        thermal_center = {'X': thermal.x, 'Y': thermal.y}
        
        if mode == 'evaluation':
            # Rayon d'évaluation: 35% du rayon du thermique pour rester dans la zone
            # de forte portance (profil gaussien: à 35% → ~78% de la force max)
            # Minimum 30m pour que le drone puisse physiquement orbiter
            orbit_radius = max(30.0, thermal.radius * 0.35)
            # Vitesse d'évaluation: vitesse min pour rester stable
            orbit_speed = max(self.UAV_data['min_airspeed'], 12.0)
        else:
            # Pour le soaring, utiliser les paramètres optimaux
            atm = _compute_atmosphere(self.params.get('working_floor', 600.0))
            flight_conditions = {
                'airspeed': self.UAV_data['min_airspeed'],
                'bank_angle': 0.0,
                'flight_path_angle': 0.0,
                'weight': 0.0,
                'grav_accel': atm['grav'],
                'air_density': atm['air_density'],
                'airspeed_dot': 0.0
            }
            soaring_params = calculate_optimal_soaring_parameters(
                self.UAV_data, thermal, flight_conditions
            )
            orbit_radius = soaring_params['optimal_radius']
            orbit_speed = soaring_params['optimal_speed']
        
        # Sens anti-horaire (négatif) pour les thermiques
        return await bridge.start_orbit_loiter(
            thermal_center, -orbit_radius, orbit_speed, 
            altitude, mode, thermal_id=getattr(thermal, 'id', None),
            thermal_obj=thermal
        )
    
    async def evaluate_thermal_orbit(self, uav_id):
        """
        Vérifier l'évaluation du thermique pour un UAV en orbite.
        
        Returns:
            dict: Résultat de l'évaluation ou None si UAV non en orbite
        """
        if uav_id >= len(self.bridges):
            return None
        
        bridge = self.bridges[uav_id]
        if not bridge.is_orbiting or bridge.orbit_mode != 'evaluation':
            return None
        
        return await bridge.evaluate_thermal_from_orbit()
    
    async def transition_to_soaring(self, uav_id, thermal):
        """
        Transitionner un UAV de l'évaluation au soaring avec rayon optimal.
        
        Args:
            uav_id (int): Index de l'UAV
            thermal: Objet Thermal
        Returns:
            bool: True si la transition a réussi
        """
        if uav_id >= len(self.bridges):
            return False
        
        bridge = self.bridges[uav_id]
        
        # Conditions atmosphériques
        atm = _compute_atmosphere(self.params.get('working_floor', 600.0))
        
        flight_conditions = {
            'airspeed': self.UAV_data.get('min_airspeed', 12.0),
            'bank_angle': 0.0,
            'flight_path_angle': 0.0,
            'weight': 0.0,
            'grav_accel': atm['grav'],
            'air_density': atm['air_density'],
            'airspeed_dot': 0.0
        }
        soaring_params = calculate_optimal_soaring_parameters(
            self.UAV_data, thermal, flight_conditions
        )
        
        return await bridge.transition_to_soaring_orbit(
            soaring_params['optimal_radius'],
            soaring_params['optimal_speed']
        )
    
    async def stop_thermal_orbit(self, uav_id):
        """
        Arrêter l'orbite thermique d'un UAV et reprendre le mode offboard.
        
        Args:
            uav_id (int): Index de l'UAV
        Returns:
            dict: Résumé de la variation d'altitude pendant l'orbite
        """
        if uav_id >= len(self.bridges):
            return None
        
        bridge = self.bridges[uav_id]
        alt_summary = bridge.get_altitude_variation()
        await bridge.stop_orbit_loiter()
        return alt_summary
    
    async def sync_orbit_position_to_simulation(self, uav_id, FLT_track, FLT_conditions):
        """
        Synchroniser la position réelle du drone en orbite vers la simulation.
        Pendant le loiter, c'est PX4 qui contrôle => lire la position réelle.
        
        Args:
            uav_id: Index de l'UAV
            FLT_track: Historique de vol
            FLT_conditions: Conditions de vol
        """
        if uav_id >= len(self.bridges):
            return
        
        bridge = self.bridges[uav_id]
        if not bridge.is_orbiting:
            return
        
        # Lire la position réelle du drone
        pos_enu = await bridge.get_current_position_enu()
        if pos_enu is None:
            return
        
        # Calculer le bearing par rapport au point précédent
        if len(FLT_track[uav_id]['X']) > 0:
            prev_x = FLT_track[uav_id]['X'][-1]
            prev_y = FLT_track[uav_id]['Y'][-1]
            dx = pos_enu['X'] - prev_x  # delta East
            dy = pos_enu['Y'] - prev_y  # delta North
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                bearing = np.arctan2(dy, dx)  # Convention math : atan2(dNorth, dEast)
            else:
                bearing = FLT_track[uav_id]['bearing'][-1] if FLT_track[uav_id]['bearing'] else 0.0
        else:
            bearing = 0.0
        
        # Mettre à jour FLT_track avec la position réelle
        FLT_track[uav_id]['X'].append(pos_enu['X'])
        FLT_track[uav_id]['Y'].append(pos_enu['Y'])
        FLT_track[uav_id]['Z'].append(pos_enu['Z'])
        FLT_track[uav_id]['bearing'].append(bearing)
        
        # Conserver le mode de vol
        current_mode = bridge.orbit_mode if bridge.orbit_mode == 'soaring' else 'glide'
        FLT_track[uav_id]['flight_mode'].append(current_mode)
        
        # Batterie: pas de consommation en glide/soaring
        if FLT_track[uav_id]['battery_capacity']:
            FLT_track[uav_id]['battery_capacity'].append(
                FLT_track[uav_id]['battery_capacity'][-1]
            )
        
        # Temps de vol
        if FLT_track[uav_id]['flight_time']:
            FLT_track[uav_id]['flight_time'].append(
                FLT_track[uav_id]['flight_time'][-1] + self.params['time_step']
            )
        else:
            FLT_track[uav_id]['flight_time'].append(self.params['time_step'])
        
        # Mettre à jour les conditions de vol
        alt_var = bridge.get_altitude_variation()
        FLT_conditions[uav_id]['flight_path_angle'] = np.arcsin(
            max(-1.0, min(1.0, alt_var['avg_climb_rate'] / max(FLT_conditions[uav_id]['airspeed'], 1.0)))
        )
        
        # Pendant le soaring, mettre à jour progressivement l'altitude d'orbite
        # pour permettre au drone de monter dans le thermique
        if bridge.orbit_mode == 'soaring':
            z_upper = self.params.get('Z_upper_bound', 1000.0)
            await bridge.update_soaring_altitude(self.params['time_step'], z_upper)
        
        return pos_enu


def generate_endurance_patrol(nUAVs, params, UAV_data):
    """
    Génère des trajectoires de patrouille en boucle pour le scénario d'endurance.
    
    Chaque UAV suit un circuit rectangulaire dans la zone de simulation, répété
    plusieurs fois. Les UAVs sont décalés dans le circuit pour une couverture
    répartie. Le test mesure combien de temps les drones tiennent en l'air en
    exploitant les thermiques.
    
    Args:
        nUAVs (int): Nombre de drones
        params (dict): Paramètres de simulation
        UAV_data (dict): Données des UAVs
        
    Returns:
        dict: Trajectoires pour chaque UAV {0: {X: [], Y: [], Z: []}, ...}
    """
    # Circuit de patrouille : rectangle à 70% de l'étendue de la zone
    x_ext = params['X_upper_bound'] * 0.7   # ±2100m avec bounds ±3000
    y_ext = params['Y_upper_bound'] * 0.7
    altitude = params['working_floor']
    
    # Points du circuit rectangulaire avec intermédiaires (8 points / boucle)
    single_loop = [
        (-x_ext, -y_ext),
        (0.0,    -y_ext),
        ( x_ext, -y_ext),
        ( x_ext,  0.0),
        ( x_ext,  y_ext),
        (0.0,     y_ext),
        (-x_ext,  y_ext),
        (-x_ext,  0.0),
    ]
    
    # Répéter le circuit pour couvrir ~1h de vol
    # Périmètre d'une boucle ≈ 4×4200m = 16.8km → à 13 m/s ≈ 20 min/boucle
    num_loops = 5
    all_waypoints = single_loop * num_loops
    
    pattern_labels = ['décalé +0', 'décalé +¼', 'décalé +½', 'décalé +¾']
    
    trajectories = {}
    pts_per_loop = len(single_loop)
    
    for u in range(nUAVs):
        # Décaler chaque UAV dans le circuit pour répartir la couverture
        offset = (u * pts_per_loop // max(nUAVs, 1)) % pts_per_loop
        shifted = all_waypoints[offset:] + all_waypoints[:offset]
        
        trajectories[u] = {
            'X': [p[0] for p in shifted],
            'Y': [p[1] for p in shifted],
            'Z': [altitude] * len(shifted)
        }
        
        label = pattern_labels[u % len(pattern_labels)]
        print(f"  UAV {u}: {len(shifted)} waypoints | patrouille {label} "
              f"({num_loops} boucles, périmètre ≈{pts_per_loop * 2 * (x_ext + y_ext) / pts_per_loop / 1000:.1f}km/boucle)")
    
    return trajectories


def generate_lawnmower_trajectories(nUAVs, params, UAV_data, fov_radius):
    """
    Génère des trajectoires LawnMower différenciées pour chaque UAV afin d'optimiser
    la couverture de la zone.
    
    Stratégie de patterns :
      - UAV 0 : 'normal'              → balayage G→D, avancement bas→haut
      - UAV 1 : 'reverse'             → même tracé parcouru en sens inverse (haut→bas)
      - UAV 2 : 'transposed'          → balayage B→H, avancement gauche→droite
      - UAV 3 : 'transposed_reverse'  → transposé parcouru en sens inverse
      - UAV 4+ : cycle sur les 4 patterns
    
    Args:
        nUAVs (int): Nombre de drones
        params (dict): Paramètres de simulation
        UAV_data (dict): Données des UAVs
        fov_radius (float): Rayon du champ de vision
        
    Returns:
        dict: Trajectoires pour chaque UAV {0: {X: [], Y: [], Z: []}, ...}
    """
    
    x_min = params['X_lower_bound']
    x_max = params['X_upper_bound']
    y_min = params['Y_lower_bound']
    y_max = params['Y_upper_bound']
    altitude = params['working_floor']
    
    coverage_area = {
        'X_min': x_min,
        'X_max': x_max,
        'Y_min': y_min,
        'Y_max': y_max,
    }
    
    # Patterns cycliques pour optimiser la couverture multi-drone
    patterns = ['normal', 'reverse', 'transposed', 'transposed_reverse']
    pattern_labels = {
        'normal': 'balayage G→D, bas→haut',
        'reverse': 'balayage G→D, haut→bas (inversé)',
        'transposed': 'balayage B→H, gauche→droite',
        'transposed_reverse': 'balayage B→H, droite→gauche (inversé)',
    }
    
    # Créer le générateur de trajectoire LawnMower
    lawnmower = LawnMowerTrajectory(params, UAV_data)
    
    # Générer les trajectoires pour chaque UAV avec un pattern différent
    trajectories = {}
    for u in range(nUAVs):
        pattern = patterns[u % len(patterns)]
        trajectory = lawnmower.generate_path(
            area_bounds=coverage_area,
            fov_radius=fov_radius,
            uav_id=u,
            num_uavs=nUAVs,
            altitude=altitude,
            pattern=pattern
        )
        trajectories[u] = trajectory
        
        print(f"  UAV {u}: {len(trajectory['X'])} waypoints | pattern: {pattern_labels[pattern]}")
    
    return trajectories


async def run_multi_uav_simulation():
    """
    Fonction principale pour simulation multi-UAV avec PX4
    Inclut tous les scénarios de test
    """
    print("="*70)
    print("SIMULATION MULTI-UAV PLANEUR AVEC THERMIQUES - PX4 SITL")
    print("TESTS DE PERFORMANCE")
    print("="*70)
    
    # ========== SÉLECTION SCÉNARIO ==========
    scenario = select_scenario()
    
    # ========== PARAMÈTRES ==========
    nUAVs = int(input("\nNombre d'UAVs à simuler (1-3): ") or "3")
    nUAVs = max(1, min(3, nUAVs))
    
    print(f"\n✓ Configuration: {scenario.value} avec {nUAVs} UAVs")
    print("\n⏳ Connexion aux drones pour obtenir les positions home...")
    
    UAV_data = dict()
    UAV_data['maximum_battery_capacity'] = 10.0
    UAV_data['desired_reserved_battery_capacity'] = UAV_data['maximum_battery_capacity'] * 0.2
    UAV_data['empty_weight'] = 1.6
    UAV_data['max_power_consumption'] = 775.0
    UAV_data['energy_conversion_efficiency'] = 0.6
    UAV_data['propeller_efficiency'] = 0.75
    UAV_data['wing_area'] = 0.5
    UAV_data['wing_aspect_ratio'] = 15.7
    UAV_data['oswald_eff_ratio'] = 0.85
    UAV_data['zero_lift_drag'] = 0.0107
    UAV_data['max_airspeed'] = 30.0
    UAV_data['min_airspeed'] = 8.0
    UAV_data['max_turn_rate'] = 0.7
    
    params = dict()
    params['working_floor'] = 600.0
    params['X_lower_bound'] = -3000.0
    params['X_upper_bound'] = 3000.0
    params['Y_lower_bound'] = -3000.0
    params['Y_upper_bound'] = 3000.0
    params['Z_lower_bound'] = 200.0
    params['Z_upper_bound'] = 1000.0
    params['current_simulation_time'] = 0.0
    params['time_step'] = 1.1  # Time step cible pour synchronisation temps réel (secondes)
    params['target_real_time_per_iteration'] = 1.1  # Temps réel cible par itération
    # Steps pour mode glide (génère moins de candidats, on peut augmenter)
    params['bearing_step_glide'] = 5
    params['speed_step_glide'] = 5
    # Steps pour mode engine (génère plus de candidats, on garde des valeurs plus basses)
    params['bearing_step_engine'] = 4
    params['speed_step_engine'] = 3
    params['safe_distance'] = 30.0
    params['horizon_length'] = 100.0
    params['adaptive_resolution'] = True  # Ajuster automatiquement la résolution si trop lent
    # Coefficient alpha (>1) pour augmenter la priorité d'un critère de décision
    # En mode glide : augmente la priorité de minimiser la descente (C_sink)
    # En mode engine : augmente la priorité de minimiser la consommation (C_energy)
    params['alpha'] = 2  # Valeur: entre 1.0 et 3.0
    
    # Initialiser métriques de performance
    metrics = PerformanceMetrics(
        scenario_name=scenario.value,
        num_uavs=nUAVs
    )
    
    # ========== CONNEXION MULTI-UAV PX4 POUR OBTENIR LES POSITIONS HOME ==========
    controller = MultiUAVController(nUAVs, params, UAV_data)
    
    try:
        # Initialiser tous les UAVs d'abord pour obtenir leurs positions home
        await controller.initialize_all_uavs()
        
        # Calculer le centre des positions home pour l'origine des scénarios
        home_positions = []
        for u in range(nUAVs):
            lat0 = controller.bridges[0].simulation_origin['lat']
            lon0 = controller.bridges[0].simulation_origin['lon']
            h0 = controller.bridges[0].simulation_origin['alt']
            
            lat = controller.bridges[u].home_position.latitude_deg
            lon = controller.bridges[u].home_position.longitude_deg
            alt = controller.bridges[u].home_position.absolute_altitude_m
            
            # Convertir en ENU local
            x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, h0)
            home_positions.append({'X': x, 'Y': y, 'Z': z, 'bearing': 0.0})
        
        print(f"\n✓ Positions home récupérées pour {nUAVs} UAVs")
        for u, pos in enumerate(home_positions):
            print(f"  UAV {u}: ({pos['X']:.1f}, {pos['Y']:.1f}, {pos['Z']:.1f})")
        
        # Calculer le centroïde (centre de la zone, maintenant centrée sur home)
        center_x = 0.0
        center_y = 0.0
        center_z = 400.0  # 400m au-dessus
        
        # ========== ROS2 THERMAL BRIDGE (Gazebo ↔ Algorithm) ==========
        # Start the bridge BEFORE scenario generation so that thermals created
        # by the ROS2 thermal_generator_node (running in Gazebo) are received
        # and converted into Thermal objects for the planning algorithm.
        #
        # Launch the generator first:
        #   ros2 launch autosoaring_pkg autosoaring_launch.py mode:=generator
        #
        # The bridge subscribes to /thermal_snapshot and populates active_thermals
        # and thermal_map automatically.
        sim_origin_lat = controller.bridges[0].simulation_origin['lat']
        sim_origin_lon = controller.bridges[0].simulation_origin['lon']
        thermal_bridge = ThermalROSBridge(origin_lat=sim_origin_lat, origin_lon=sim_origin_lon)
        thermal_bridge.start()
        
        # ========== GÉNÉRATION SCÉNARIO BASÉE SUR LES POSITIONS HOME ==========
        scenario_gen = ScenarioGenerator()
        surveillance_objects = None
        # active_thermals and thermal_map are now managed by the ROS2 bridge
        # The bridge continuously updates them from Gazebo thermal_generator_node
        active_thermals = thermal_bridge.get_active_thermals()
        thermal_map = thermal_bridge.get_thermal_map()
        
        if scenario == TestScenario.PRELIMINARY_COLLISION:
            # Les drones partent de leurs positions home et convergent vers le centre
            start_positions, end_position, obstacles = scenario_gen.generate_preliminary_collision_scenario(nUAVs, params, home_positions)
            params['obstacles'] = obstacles
            allow_glide = True
            
        elif scenario == TestScenario.TRAJECTORY_OPTIMAL_POWERED:
            # Positions de départ = positions home, destination éloignée
            allow_glide = False
            start_positions, end_position, obstacles = scenario_gen.generate_trajectory_optimal_scenario(nUAVs, params, home_positions, allow_glide)
            params['obstacles'] = obstacles
            
            print(f"\n✓ Scénario trajectoire optimale (motorisé) vers ({end_position['X']:.0f}, {end_position['Y']:.0f})")
            
        elif scenario == TestScenario.TRAJECTORY_OPTIMAL_GLIDE:
            # Positions de départ = positions home, destination éloignée avec planage
            allow_glide = True
            start_positions, end_position, obstacles = scenario_gen.generate_trajectory_optimal_scenario(nUAVs, params, home_positions, allow_glide)
            params['obstacles'] = obstacles
            # Thermiques exclusivement depuis Gazebo via ROS2 bridge (thermal_generator_node)
            print("⏳ Attente des thermiques Gazebo via ROS2 bridge...")
            if thermal_bridge.wait_for_thermals(timeout=60.0):
                active_thermals = thermal_bridge.get_active_thermals()
                print(f'✓ {len(active_thermals)} thermiques reçues de Gazebo')
            else:
                print('⚠️  Aucune thermique reçue de Gazebo — la simulation continuera sans thermiques')
                print('    Vérifiez que thermal_generator_node tourne : ros2 launch autosoaring_pkg autosoaring_launch.py mode:=generator')
            print(f"\n✓ Scénario trajectoire optimale (planeur) vers ({end_position['X']:.0f}, {end_position['Y']:.0f})")
            
        elif scenario == TestScenario.ENDURANCE:
            # ===== SCÉNARIO D'ENDURANCE =====
            # Les drones patrouillent en boucle dans la zone et exploitent les
            # thermiques pour rester en l'air le plus longtemps possible.
            # Pas de destination unique — le test mesure le temps de vol total,
            # le ratio soaring/glide, et la consommation batterie.
            obstacles = generate_random_obstacles(3, params)
            params['obstacles'] = obstacles
            # Thermiques exclusivement depuis Gazebo via ROS2 bridge
            print("⏳ Attente des thermiques Gazebo via ROS2 bridge...")
            if thermal_bridge.wait_for_thermals(timeout=60.0):
                active_thermals = thermal_bridge.get_active_thermals()
                print(f'✓ {len(active_thermals)} thermiques reçues de Gazebo')
            else:
                print('⚠️  Aucune thermique reçue de Gazebo — la simulation continuera sans thermiques')
                print('    Vérifiez que thermal_generator_node tourne : ros2 launch autosoaring_pkg autosoaring_launch.py mode:=generator')
            metrics.thermals_generated = len(active_thermals)
            allow_glide = True
            start_positions = {u: home_positions[u] for u in range(nUAVs)}
            # Pas de end_position pour endurance : les trajectoires sont des patrouilles en boucle
            end_position = None
            print(f"\n✓ Scénario endurance: {len(active_thermals)} thermiques, patrouille en boucle")
        
        elif scenario == TestScenario.COVERAGE:
            mission_duration = 1200  # 20 minutes
            surveillance_objects = scenario_gen.generate_coverage_scenario(nUAVs, params, mission_duration)
            metrics.objects_total = len(surveillance_objects)
            obstacles = generate_random_obstacles(5, params)
            params['obstacles'] = obstacles
            allow_glide = True
            # Thermiques exclusivement depuis Gazebo via ROS2 bridge
            print("⏳ Attente des thermiques Gazebo via ROS2 bridge...")
            if thermal_bridge.wait_for_thermals(timeout=60.0):
                active_thermals = thermal_bridge.get_active_thermals()
                print(f'✓ {len(active_thermals)} thermiques reçues de Gazebo')
            else:
                print('⚠️  Aucune thermique reçue de Gazebo — la simulation continuera sans thermiques')
                print('    Vérifiez que thermal_generator_node tourne : ros2 launch autosoaring_pkg autosoaring_launch.py mode:=generator')
            start_positions = {u: home_positions[u] for u in range(nUAVs)}
            print(f"\n✓ Scénario couverture: {len(surveillance_objects)} objets à surveiller")
        
        
        # ========== CALCULS ATMOSPHÉRIQUES ==========
        atm = _compute_atmosphere(params['working_floor'])
        grav_accel = atm['grav']
        air_density = atm['air_density']
        
        # ========== INITIALISATION UAVs ==========
        # Champ de vision pour détection d'objets et espacement trajectoires
        fov_radius = 150.0
        
        FLT_track = {k: {} for k in range(nUAVs)}
        FLT_track_keys = ['X', 'Y', 'Z', 'bearing', 'battery_capacity', 'flight_time', 
                          'flight_mode', 'in_evaluation', 'current_thermal_id', 'soaring_start_time']
        FLT_conditions = {k: {} for k in range(nUAVs)}
        END_WPs = {k: {} for k in range(nUAVs)}
        WPs_keys = ['X', 'Y', 'Z']
        soar_keys = ['X', 'Y', 'Z', 'bearing', 'flight_path_angle', 'bank_angle']
        GOAL_WPs = {k: {} for k in range(nUAVs)}
        EVAL_WPs = {k: {} for k in range(nUAVs)}
        SOAR_WPs = {k: {} for k in range(nUAVs)}
        
        print(f"\nInitialisation des trajectoires pour {nUAVs} UAVs...")
        
        # Initialiser tous les dictionnaires pour tous les UAVs
        for u in range(nUAVs):
            # Initialiser les dictionnaires
            FLT_track[u] = dict()
            for keys in FLT_track_keys:
                FLT_track[u][keys] = []
            for keys in WPs_keys:
                END_WPs[u][keys] = []
                GOAL_WPs[u][keys] = []
                EVAL_WPs[u][keys] = []
            for keys in soar_keys:
                SOAR_WPs[u][keys] = []

            FLT_conditions[u] = dict()
            FLT_conditions[u]['airspeed'] = 13.0
            FLT_conditions[u]['weight'] = 0.0
            FLT_conditions[u]['flight_path_angle'] = 0.0
            FLT_conditions[u]['grav_accel'] = grav_accel
            FLT_conditions[u]['bank_angle'] = 0.0
            FLT_conditions[u]['airspeed_dot'] = 0.0
            FLT_conditions[u]['air_density'] = air_density
            FLT_conditions[u]['battery_capacity'] = UAV_data['maximum_battery_capacity']

            if scenario not in (TestScenario.COVERAGE, TestScenario.ENDURANCE):
                END_WPs[u]['X'].append(end_position['X'])
                END_WPs[u]['Y'].append(end_position['Y'])
                END_WPs[u]['Z'].append(end_position['Z'])

            # Utiliser les positions de départ du scénario (basées sur home positions)
            initial_x = start_positions[u]['X']
            initial_y = start_positions[u]['Y']
            initial_z = start_positions[u]['Z'] + params['working_floor']
            initial_bearing = start_positions[u]['bearing']
            print(f"UAV {u} départ (E,N,U): ({initial_x:.1f}, {initial_y:.1f}, {initial_z:.1f}) cap: {np.degrees(initial_bearing):.0f}°")
            
            FLT_track[u]['X'].append(initial_x)
            FLT_track[u]['Y'].append(initial_y)
            FLT_track[u]['Z'].append(initial_z)
            FLT_track[u]['bearing'].append(initial_bearing)
            FLT_track[u]['battery_capacity'].append(UAV_data['maximum_battery_capacity'])
            FLT_track[u]['flight_time'].append(0.0)
            FLT_track[u]['flight_mode'].append('glide' if allow_glide else 'engine')
            FLT_track[u]['in_evaluation'] = False
            FLT_track[u]['current_thermal_id'] = None
            FLT_track[u]['soaring_start_time'] = None
            FLT_track[u]['seeking_thermal'] = False
            FLT_track[u]['seeking_thermal_id'] = None
            FLT_track[u]['seeking_wp_idx'] = None  # Index du WP de secours inséré dans GOAL_WPs
            FLT_track[u]['visited_thermals'] = set()  # Thermiques déjà visitées par cet UAV
            FLT_track[u]['thermal_exit_cooldown'] = {}  # {thermal_id: exit_time} cooldown après sortie
            FLT_track[u]['thermal_exploitation_log'] = []  # [{'thermal_id', 'entry_time', 'exit_time'}]
            FLT_track[u]['thermals_detected_count'] = 0
            FLT_track[u]['collisions_avoided_count'] = 0

        # Fonction pour générer et évaluer la trajectoire d'un UAV
        async def generate_trajectory_for_uav(u):
            """Génération et évaluation de trajectoire pour un UAV"""
            startPoint = dict()
            startPoint['X'] = FLT_track[u]['X'][-1]
            startPoint['Y'] = FLT_track[u]['Y'][-1]
            startPoint['Z'] = FLT_track[u]['Z'][-1]
            startPoint['bearing'] = FLT_track[u]['bearing'][-1]
            
            # Exécuter dans un thread pour ne pas bloquer
            loop = asyncio.get_event_loop()
            
            def compute_trajectory():
                evaluator = TrajectoryEvaluator(params, UAV_data, FLT_conditions[u])
                trajectoires = generate_all_trajectories(startPoint, END_WPs[u], params, UAV_data, obstacles)
                optimal_trajectoires = evaluator.evaluate_trajectories(trajectoires)
                return optimal_trajectoires
            
            # Exécuter en parallèle dans un executor pour ne pas bloquer la boucle asyncio
            optimal_trajectoires = await loop.run_in_executor(None, compute_trajectory)
            
            GOAL_WPs[u]['X'] = optimal_trajectoires['X']
            GOAL_WPs[u]['Y'] = optimal_trajectoires['Y']
            GOAL_WPs[u]['Z'] = optimal_trajectoires['Z']
            
            return u
        
        # Générer toutes les trajectoires en parallèle
        print("\n⚡ Génération des trajectoires en parallèle...")
        traj_start = time.perf_counter()
        
        # Pour le scénario COVERAGE, utiliser les trajectoires LawnMower
        if scenario == TestScenario.COVERAGE:
            print("\n🔷 Mode COVERAGE détecté - Génération trajectoires LawnMower...")
            lawnmower_trajectories = generate_lawnmower_trajectories(
                nUAVs, params, UAV_data, fov_radius
            )
            
            # Assigner les trajectoires LawnMower aux GOAL_WPs
            for u in range(nUAVs):
                GOAL_WPs[u]['X'] = lawnmower_trajectories[u]['X']
                GOAL_WPs[u]['Y'] = lawnmower_trajectories[u]['Y']
                GOAL_WPs[u]['Z'] = lawnmower_trajectories[u]['Z']
            
            completed_uavs = list(range(nUAVs))
        
        elif scenario == TestScenario.ENDURANCE:
            print("\n🔷 Mode ENDURANCE détecté - Génération trajectoires de patrouille...")
            patrol_trajectories = generate_endurance_patrol(nUAVs, params, UAV_data)
            
            for u in range(nUAVs):
                GOAL_WPs[u]['X'] = patrol_trajectories[u]['X']
                GOAL_WPs[u]['Y'] = patrol_trajectories[u]['Y']
                GOAL_WPs[u]['Z'] = patrol_trajectories[u]['Z']
            
            completed_uavs = list(range(nUAVs))
        
        else:
            # Pour les autres scénarios, utiliser la génération dynamique
            trajectory_tasks = [generate_trajectory_for_uav(u) for u in range(nUAVs)]
            completed_uavs = await asyncio.gather(*trajectory_tasks)
        
        traj_end = time.perf_counter()
        traj_time = traj_end - traj_start
        
        print(f"✓ Trajectoires générées pour {len(completed_uavs)} UAVs en {traj_time:.2f}s")
        for u in completed_uavs:
            print(f"  UAV {u}: {len(GOAL_WPs[u]['X'])} waypoints")
        
        current_wp_indices = {u: 1 for u in range(nUAVs)}
        current_eval_wp_indices = {u: 1 for u in range(nUAVs)}
        current_soar_wp_indices = {u: 1 for u in range(nUAVs)}
        
        # Décoller tous les UAVs
        await controller.arm_and_takeoff_all()
        
        #await controller.set_altitude_all(400.0)
        
        # Démarrer le mode offboard pour tous
        await controller.start_offboard_all()
        
        print("\n" + "="*70)
        print(f"DÉBUT DE LA SIMULATION - {scenario.value}")
        print("="*70)
        
        # ========== BOUCLE PRINCIPALE ==========
        iteration = 0
        if scenario == TestScenario.ENDURANCE:
            max_iterations = 5000   # ~73 min : test d'endurance long
        elif scenario == TestScenario.COVERAGE:
            max_iterations = 5000   # ~73 min : couverture longue
        else:
            max_iterations = 2000   # ~37 min
        
        # Compteurs de boucles de patrouille pour endurance
        patrol_loops = {u: 0 for u in range(nUAVs)}
        
        total_decision_time = 0
        decision_calls = 0
        max_decision_time = 0
        
        # Horloge temps réel (pour statistiques seulement)
        real_time_start = time.perf_counter()
        
        print(f"\n⚙️  Configuration temps réel:")
        print(f"   Time step cible: {params['time_step']:.2f}s")
        print(f"   Bearing steps - Glide: {params['bearing_step_glide']}, Engine: {params['bearing_step_engine']}")
        print(f"   Speed steps - Glide: {params['speed_step_glide']}, Engine: {params['speed_step_engine']}")
        print(f"   Résolution adaptive: {'Activée' if params['adaptive_resolution'] else 'Désactivée'}")
        print("   → Objectif: calcul en ~{:.0f}ms pour réactivité maximale\n".format(params['target_real_time_per_iteration']*1000))
        
        while iteration < max_iterations:
            # ===== ENDURANCE : boucler la patrouille quand WPs terminés =====
            if scenario == TestScenario.ENDURANCE:
                for u in range(nUAVs):
                    if controller.bridges[u].is_landing:
                        continue
                    if current_wp_indices[u] >= len(GOAL_WPs[u]['X']):
                        current_wp_indices[u] = 0
                        patrol_loops[u] += 1
                        print(f"  ↻ UAV {u}: Patrouille boucle {patrol_loops[u]} terminée → redémarrage")
            
            # Vérifier si tous les drones actifs ont atterri → fin de simulation
            if all(controller.bridges[u].is_landing for u in range(nUAVs)):
                print("\n✓ Tous les UAVs ont atterri — fin de simulation.")
                break
            
            # Vérifier si tous les UAVs actifs ont terminé (sauf endurance = boucle infinie)
            if scenario != TestScenario.ENDURANCE and all(
                controller.bridges[u].is_landing or current_wp_indices[u] >= len(GOAL_WPs[u]['X'])
                for u in range(nUAVs)
            ):
                print("\n✓ Tous les UAVs ont atteint leurs objectifs!")
                break
            
            # Synchroniser les positions réelles MAVSDK → FLT_track
            # avant l'algorithme de décision pour qu'il travaille avec les vraies positions
            await controller.sync_all_positions_from_mavsdk(FLT_track, FLT_conditions)
            
            # Mesurer le temps de décision de l'algorithme
            algo_start_time = time.perf_counter()
            
            # Mise à jour de la simulation (navigation waypoints uniquement)
            FLT_track, FLT_conditions, current_wp_indices = gotoWaypointMulti(
                FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data,
                current_wp_indices
            )
            
            algo_end_time = time.perf_counter()
            algo_execution_time = algo_end_time - algo_start_time
            
            total_decision_time += algo_execution_time
            decision_calls += 1
            if algo_execution_time > max_decision_time:
                max_decision_time = algo_execution_time
            
            # SYNCHRONISATION TEMPS RÉEL INTELLIGENTE
            # Si le calcul est trop lent, réduire la résolution adaptativement
            if params['adaptive_resolution'] and algo_execution_time > params['target_real_time_per_iteration'] * 2:
                if params['bearing_step_glide'] > 4:
                    params['bearing_step_glide'] = max(4, params['bearing_step_glide'] - 1)
                    print(f"⚠️  Calcul trop lent ({algo_execution_time:.2f}s) - Réduction bearing_step_glide à {params['bearing_step_glide']}")
                #if params['bearing_step_engine'] > 3:
                #    params['bearing_step_engine'] = max(3, params['bearing_step_engine'] - 1)
                #    print(f"⚠️  Calcul trop lent - Réduction bearing_step_engine à {params['bearing_step_engine']}")
                if params['speed_step_glide'] > 3:
                    params['speed_step_glide'] = max(3, params['speed_step_glide'] - 1)
                    print(f"⚠️  Calcul trop lent - Réduction speed_step_glide à {params['speed_step_glide']}")
                #if params['speed_step_engine'] > 2:
                #    params['speed_step_engine'] = max(2, params['speed_step_engine'] - 1)
                #    print(f"⚠️  Calcul trop lent - Réduction speed_step_engine à {params['speed_step_engine']}")
            
            # Mettre à jour le temps de simulation
            params['current_simulation_time'] += params['time_step']
            current_time = params['current_simulation_time']
            
            # ========== GESTION THERMIQUES, ORBITES ET SECOURS PAR UAV ==========
            # Refresh active_thermals from the ROS2 bridge (picks up new/expired thermals from Gazebo)
            if thermal_bridge.has_thermals():
                active_thermals = thermal_bridge.get_active_thermals()
            has_thermals = bool(thermal_map and active_thermals)
            
            # --- Affichage périodique des thermiques ROS reçues ---
            if iteration == 0 or (iteration % 100 == 0 and has_thermals):
                print(f"\n{'='*60}")
                print(f"🌡️  THERMIQUES ROS ACTIVES (iter {iteration}, t_sim={current_time:.1f}s)")
                print(f"{'='*60}")
                if not active_thermals:
                    print("  ❌ Aucune thermique reçue du bridge ROS2")
                    print(f"     Bridge update: {thermal_bridge.seconds_since_last_update():.1f}s ago")
                else:
                    print(f"  Nombre de thermiques actives: {len(active_thermals)}")
                    for tid, th in active_thermals.items():
                        th_obj = th
                        is_ros = getattr(th_obj, 'ros_managed', False)
                        print(f"  [{tid}] pos=({th_obj.x:.0f}, {th_obj.y:.0f}) "
                              f"r={th_obj.radius:.0f}m strength={th_obj.strength:.1f}m/s "
                              f"{'[ROS]' if is_ros else '[local]'}")
                    # Distances UAV → thermique la plus proche
                    for u in range(nUAVs):
                        if FLT_track[u]['X']:
                            ux, uy = FLT_track[u]['X'][-1], FLT_track[u]['Y'][-1]
                            min_dist = float('inf')
                            nearest_tid = None
                            for tid, th in active_thermals.items():
                                th_obj = th
                                d = np.sqrt((ux - th_obj.x)**2 + (uy - th_obj.y)**2)
                                if d < min_dist:
                                    min_dist = d
                                    nearest_tid = tid
                            if nearest_tid is not None:
                                th_obj = active_thermals[nearest_tid]
                                inside = "DANS" if min_dist <= th_obj.radius else "hors"
                                print(f"  UAV {u}: ({ux:.0f},{uy:.0f}) → thermique {nearest_tid} "
                                      f"dist={min_dist:.0f}m ({inside} rayon={th_obj.radius:.0f}m)")
                print(f"{'='*60}\n")
            z_lower = params['Z_lower_bound']
            low_alt_margin = 100.0
            # Endurance : seuil proactif pour chercher des thermiques
            # Les drones cherchent dès qu'ils passent sous le working_floor
            endurance_seeking = (scenario == TestScenario.ENDURANCE)
            
            for u in range(nUAVs):
                bridge = controller.bridges[u]
                # Ignorer les drones en atterrissage/atterris
                if bridge.is_landing:
                    continue
                thermal_id = FLT_track[u]['current_thermal_id']
                flight_mode = FLT_track[u]['flight_mode'][-1] if FLT_track[u]['flight_mode'] else 'glide'
                
                if not FLT_track[u]['X']:
                    continue
                
                # --- ORBITE D'ÉVALUATION : vérifier la variation d'altitude ---
                if bridge.is_orbiting and bridge.orbit_mode == 'evaluation':
                    eval_result = await controller.evaluate_thermal_orbit(u)
                    if eval_result and eval_result['evaluation_complete']:
                        if eval_result['thermal_viable']:
                            print(f"\n✅ UAV {u}: Thermique {thermal_id} viable "
                                  f"(climb={eval_result['climb_rate']:.2f}m/s) → Transition soaring MAVSDK")
                            if thermal_map and thermal_id is not None:
                                thermal_map.change_thermal_status(
                                    thermal_id, evaluated=True, 
                                    alt_gain=eval_result.get('altitude_gain', 0)
                                )
                            if thermal_id is not None and thermal_id in active_thermals:
                                thermal_obj = active_thermals[thermal_id]
                                await controller.transition_to_soaring(u, thermal_obj)
                            FLT_track[u]['in_evaluation'] = False
                            # Nettoyer l'obstacle d'évaluation
                            _remove_evaluation_obstacles(params, u, thermal_id)
                            FLT_track[u]['flight_mode'].append('soaring')
                            FLT_track[u]['soaring_start_time'] = current_time
                            if thermal_id is not None:
                                FLT_track[u]['thermal_exploitation_log'].append({
                                    'thermal_id': thermal_id,
                                    'entry_time': current_time,
                                    'exit_time': None,
                                })
                        else:
                            print(f"\n❌ UAV {u}: Thermique {thermal_id} non viable "
                                  f"(climb={eval_result['climb_rate']:.2f}m/s) → Reprise trajectoire")
                            if thermal_map and thermal_id is not None:
                                thermal_map.change_thermal_status(
                                    thermal_id, evaluated=True, alt_gain=0
                                )
                            await controller.stop_thermal_orbit(u)
                            FLT_track[u]['in_evaluation'] = False
                            # Nettoyer l'obstacle d'évaluation
                            _remove_evaluation_obstacles(params, u, thermal_id)
                            # Marquer comme visitée pour ne pas retenter
                            if thermal_id is not None:
                                FLT_track[u]['visited_thermals'].add(thermal_id)
                            FLT_track[u]['current_thermal_id'] = None
                            FLT_track[u]['flight_mode'].append('glide')
                    else:
                        await controller.sync_orbit_position_to_simulation(u, FLT_track, FLT_conditions)
                    continue
                
                # --- ORBITE SOARING : vérifier conditions de sortie ---
                if bridge.is_orbiting and bridge.orbit_mode == 'soaring':
                    await controller.sync_orbit_position_to_simulation(u, FLT_track, FLT_conditions)
                    
                    alt_var = bridge.get_altitude_variation()
                    current_alt = FLT_track[u]['Z'][-1]
                    # Safeguard : si soaring_start_time n'est pas défini alors
                    # qu'on est en soaring, l'initialiser maintenant
                    if FLT_track[u]['soaring_start_time'] is None:
                        FLT_track[u]['soaring_start_time'] = current_time
                        print(f"⚠️  UAV {u}: soaring_start_time était None en soaring — initialisé à {current_time:.1f}s")
                    soaring_duration = current_time - FLT_track[u]['soaring_start_time']
                    
                    should_exit_soaring = False
                    exit_reason = ""
                    
                    # 1. Thermique disparue de la liste ROS (retirée par le générateur)
                    if thermal_id is not None and thermal_id not in active_thermals:
                        should_exit_soaring = True
                        exit_reason = f"thermique {thermal_id} retirée (disparue des actives)"
                    # 2. Thermique inactive (durée de vie écoulée)
                    elif thermal_id is not None and thermal_id in active_thermals:
                        t_obj = active_thermals[thermal_id]
                        if hasattr(t_obj, 'is_active') and not t_obj.is_active(current_time):
                            should_exit_soaring = True
                            exit_reason = "thermique inactive (durée de vie écoulée)"
                    # 3. Altitude plafond atteinte
                    if not should_exit_soaring and current_alt >= params['Z_upper_bound'] * 0.95:
                        should_exit_soaring = True
                        exit_reason = f"altitude plafond ({current_alt:.0f}m ≥ {params['Z_upper_bound']*0.95:.0f}m)"
                    # 4. Thermique épuisée : climb rate trop faible après un temps suffisant
                    #    Condition renforcée : durée > 45s ET perte nette d'altitude (climb < 0)
                    #    OU durée > 60s ET gain négligeable (climb < 0.1)
                    if not should_exit_soaring and alt_var['duration'] > 45 and alt_var['avg_climb_rate'] < 0:
                        should_exit_soaring = True
                        exit_reason = f"perte d'altitude nette après {alt_var['duration']:.0f}s (climb={alt_var['avg_climb_rate']:.2f}m/s)"
                    elif not should_exit_soaring and alt_var['duration'] > 60 and alt_var['avg_climb_rate'] < 0.1:
                        should_exit_soaring = True
                        exit_reason = f"gain négligeable après {alt_var['duration']:.0f}s (climb={alt_var['avg_climb_rate']:.2f}m/s)"
                    # 5. Durée max de soaring
                    if not should_exit_soaring and soaring_duration > 300:
                        should_exit_soaring = True
                        exit_reason = f"durée max ({soaring_duration:.0f}s)"
                    
                    if should_exit_soaring:
                        print(f"\n🔄 UAV {u}: Sortie soaring MAVSDK ({exit_reason})")
                        alt_summary = await controller.stop_thermal_orbit(u)
                        FLT_track[u]['flight_mode'].append('glide')
                        FLT_track[u]['soaring_start_time'] = None
                        log_entries = FLT_track[u].get('thermal_exploitation_log', [])
                        if log_entries and log_entries[-1].get('exit_time') is None:
                            log_entries[-1]['exit_time'] = current_time
                        
                        current_pos = {
                            'X': FLT_track[u]['X'][-1],
                            'Y': FLT_track[u]['Y'][-1],
                            'Z': FLT_track[u]['Z'][-1]
                        }
                        exit_thermal = None
                        if thermal_id is not None and thermal_map:
                            exit_thermal = thermal_map.get_thermal_obj(thermal_id)
                        current_wp_indices[u] = find_nearest_waypoint(
                            current_pos, GOAL_WPs[u], params['obstacles'],
                            exit_thermal, current_wp_indices[u]
                        )
                        # Marquer la thermique comme visitée + enregistrer cooldown
                        if thermal_id is not None:
                            FLT_track[u]['visited_thermals'].add(thermal_id)
                            FLT_track[u]['thermal_exit_cooldown'][thermal_id] = current_time
                        FLT_track[u]['current_thermal_id'] = None
                        
                        if alt_summary:
                            print(f"  Bilan orbite: Δalt={alt_summary['total_change']:+.1f}m, "
                                  f"taux moyen={alt_summary['avg_climb_rate']:.2f}m/s")
                    continue
                
                # --- DÉTECTION THERMIQUE (UAV libre, pas en orbite) ---
                if has_thermals and not FLT_track[u]['in_evaluation'] and flight_mode != 'soaring':
                    current_pos = {
                        'X': FLT_track[u]['X'][-1],
                        'Y': FLT_track[u]['Y'][-1],
                        'Z': FLT_track[u]['Z'][-1]
                    }
                    
                    # Réinitialiser les thermiques visitées dont le cooldown a expiré (60s)
                    # Ne PAS faire un reset global basé sur l'altitude (causait des boucles
                    # sortie/re-entrée immédiates quand le drone est encore dans le rayon)
                    THERMAL_COOLDOWN_SECONDS = 60.0
                    expired_cooldowns = [
                        tid for tid, exit_t in FLT_track[u]['thermal_exit_cooldown'].items()
                        if current_time - exit_t > THERMAL_COOLDOWN_SECONDS
                    ]
                    for tid in expired_cooldowns:
                        FLT_track[u]['visited_thermals'].discard(tid)
                        del FLT_track[u]['thermal_exit_cooldown'][tid]
                        if tid in active_thermals:
                            print(f"🔓 UAV {u}: Cooldown expiré pour thermique {tid} — disponible à nouveau")
                    # Reset uniquement les thermiques sans cooldown actif si altitude basse
                    if current_pos['Z'] < params['working_floor'] and FLT_track[u]['visited_thermals']:
                        cooldown_ids = set(FLT_track[u]['thermal_exit_cooldown'].keys())
                        resettable = FLT_track[u]['visited_thermals'] - cooldown_ids
                        if resettable:
                            print(f"↓ UAV {u}: Altitude sous working_floor ({current_pos['Z']:.0f}m) → Reset {len(resettable)} thermique(s) (hors cooldown)")
                            FLT_track[u]['visited_thermals'] -= resettable
                    
                    detected_thermal_id = detect_thermal_at_position(
                        current_pos, active_thermals, current_time
                    )
                    
                    # Log de détection détaillé (toutes les 50 itérations)
                    if iteration % 50 == 0 and detected_thermal_id is None:
                        # Trouver la thermique la plus proche pour le debug
                        min_d = float('inf')
                        closest_tid = None
                        for tid, th in active_thermals.items():
                            th_obj = th
                            d = np.sqrt((current_pos['X'] - th_obj.x)**2 + (current_pos['Y'] - th_obj.y)**2)
                            if d < min_d:
                                min_d = d
                                closest_tid = tid
                        if closest_tid is not None:
                            th_obj = active_thermals[closest_tid]
                            is_act = th_obj.is_active(current_time)
                            lift = th_obj.get_lift_rate(min_d)
                            print(f"  [detect] UAV {u}: pas de thermique — plus proche={closest_tid} "
                                  f"dist={min_d:.0f}m (rayon={th_obj.radius:.0f}m) "
                                  f"is_active={is_act} lift@dist={lift:.2f}m/s")
                    
                    if detected_thermal_id is not None:
                        # Marquer comme physiquement détectée par un drone
                        if thermal_map:
                            thermal_map.mark_detected(detected_thermal_id)
                        FLT_track[u]['thermals_detected_count'] += 1
                        # Ignorer les thermiques déjà visitées par cet UAV
                        if detected_thermal_id in FLT_track[u]['visited_thermals']:
                            pass  # Ne pas réentrer dans cette thermique
                        else:
                            thermal_info = thermal_map.get_info(detected_thermal_id)
                            is_evaluated = thermal_info.get('evaluated', False) if thermal_info else False
                            is_viable = is_evaluated and thermal_info.get('alt_gain', 0) > 0
                            is_bad = is_evaluated and thermal_info.get('alt_gain', 0) <= 0
                            
                            if is_bad:
                                # Cas C: Thermique connue non viable → marquer visitée pour ne plus détecter
                                FLT_track[u]['visited_thermals'].add(detected_thermal_id)
                            
                            elif is_viable:
                                # Cas B: Thermique connue viable → soaring direct
                                alt_conflict = _check_thermal_altitude_conflict(
                                    u, detected_thermal_id, current_pos['Z'],
                                    controller, FLT_track, nUAVs
                                )
                                if not alt_conflict:
                                    # Annuler seeking si en cours (thermique trouvée en route vers secours)
                                    if FLT_track[u]['seeking_thermal']:
                                        print(f"  UAV {u}: Thermique {detected_thermal_id} détectée en route → annulation seeking vers {FLT_track[u]['seeking_thermal_id']}")
                                        _cancel_seeking(FLT_track[u], GOAL_WPs[u], current_wp_indices, u)
                                    thermal_obj = active_thermals[detected_thermal_id]
                                    FLT_track[u]['current_thermal_id'] = detected_thermal_id
                                    FLT_track[u]['flight_mode'].append('soaring')
                                    FLT_track[u]['soaring_start_time'] = current_time
                                    print(f"\n🔄 UAV {u}: Thermique {detected_thermal_id} connue viable → Orbite soaring MAVSDK")
                                    orbit_ok = await controller.start_thermal_orbit(u, thermal_obj, current_pos['Z'], mode='soaring')
                                    if not orbit_ok:
                                        FLT_track[u]['flight_mode'][-1] = 'glide'
                                        FLT_track[u]['soaring_start_time'] = None
                                        FLT_track[u]['current_thermal_id'] = None
                                        print(f"  UAV {u}: Échec orbite soaring, retour en glide")
                                    else:
                                        FLT_track[u]['thermal_exploitation_log'].append({
                                            'thermal_id': detected_thermal_id,
                                            'entry_time': current_time,
                                            'exit_time': None,
                                        })
                                    continue
                                else:
                                    print(f"  UAV {u}: Thermique {detected_thermal_id} occupée à altitude similaire, passage")
                            
                            else:
                                # Cas A: Thermique inconnue OU connue mais pas encore évaluée
                                # (le bridge ROS pré-remplit thermal_map avec evaluated=False)
                                # → lancer une orbite d'évaluation
                                # Annuler seeking si en cours
                                if FLT_track[u]['seeking_thermal']:
                                    print(f"  UAV {u}: Thermique {detected_thermal_id} détectée en route → annulation seeking vers {FLT_track[u]['seeking_thermal_id']}")
                                    _cancel_seeking(FLT_track[u], GOAL_WPs[u], current_wp_indices, u)
                                thermal_obj = active_thermals[detected_thermal_id]
                                # Ajouter à la carte seulement si pas déjà présente
                                if not thermal_info:
                                    thermal_map.ensure_exists(detected_thermal_id, thermal_obj, current_time)
                                
                                FLT_track[u]['in_evaluation'] = True
                                FLT_track[u]['current_thermal_id'] = detected_thermal_id
                                FLT_track[u]['evaluation_start_altitude'] = current_pos['Z']
                                
                                evaluation_obstacle = {
                                    'X': thermal_obj.x, 'Y': thermal_obj.y,
                                    'radius': thermal_obj.radius,
                                    'type': 'evaluation_zone', 'uav_id': u
                                }
                                eval_poly = convert_cylindrical_obstacles_to_polygons([evaluation_obstacle])
                                params['obstacles'].append({
                                    'vertices': eval_poly[0], 'uav_id': u,
                                    'thermal_id': detected_thermal_id, 'type': 'evaluation_zone'
                                })
                                
                                eval_label = "non évaluée" if thermal_info else "nouvelle"
                                print(f"\n🌀 UAV {u}: Thermique {detected_thermal_id} ({eval_label}) détectée → Orbite d'évaluation MAVSDK")
                                orbit_ok = await controller.start_thermal_orbit(u, thermal_obj, current_pos['Z'], mode='evaluation')
                                if not orbit_ok:
                                    FLT_track[u]['in_evaluation'] = False
                                    FLT_track[u]['current_thermal_id'] = None
                                    _remove_evaluation_obstacles(params, u, detected_thermal_id)
                                    print(f"  UAV {u}: Échec orbite évaluation, retour en glide")
                                continue
                
                # --- SECOURS BASSE ALTITUDE (UAV libre, pas en orbite, pas en soaring) ---
                if has_thermals and flight_mode != 'soaring':
                    # Vérifier seeking en cours
                    if FLT_track[u]['seeking_thermal']:
                        target_tid = FLT_track[u]['seeking_thermal_id']
                        seeking_cancelled = False
                        
                        # 1. Thermique cible disparue → annuler seeking
                        if target_tid is None or target_tid not in active_thermals:
                            print(f"⚠️  UAV {u}: Thermique de secours {target_tid} disparue → Annulation seeking")
                            _cancel_seeking(FLT_track[u], GOAL_WPs[u], current_wp_indices, u)
                            seeking_cancelled = True
                        
                        # 2. Altitude remontée → annuler seeking
                        # Endurance : annuler quand au-dessus du working_floor + marge
                        # Autres : annuler quand au-dessus de z_lower + hystérésis
                        elif endurance_seeking and FLT_track[u]['Z'][-1] > params['working_floor'] + 50:
                            print(f"↑ UAV {u}: Altitude remontée ({FLT_track[u]['Z'][-1]:.0f}m > "
                                  f"{params['working_floor'] + 50:.0f}m) → Annulation seeking")
                            _cancel_seeking(FLT_track[u], GOAL_WPs[u], current_wp_indices, u)
                            seeking_cancelled = True
                        elif not endurance_seeking and FLT_track[u]['Z'][-1] > z_lower + low_alt_margin * 2:
                            print(f"↑ UAV {u}: Altitude remontée ({FLT_track[u]['Z'][-1]:.0f}m > "
                                  f"{z_lower + low_alt_margin * 2:.0f}m) → Annulation seeking")
                            _cancel_seeking(FLT_track[u], GOAL_WPs[u], current_wp_indices, u)
                            seeking_cancelled = True
                        
                        # 3. Vérifier arrivée à la thermique
                        else:
                            t_obj = active_thermals[target_tid]
                            dist_to_thermal = np.sqrt(
                                (FLT_track[u]['X'][-1] - t_obj.x)**2 + 
                                (FLT_track[u]['Y'][-1] - t_obj.y)**2
                            )
                            if dist_to_thermal <= t_obj.radius:
                                # Vérifier si ce thermique a déjà été évalué non viable
                                thermal_already_evaluated_bad = False
                                if thermal_map:
                                    t_info = thermal_map.get_info(target_tid)
                                    if t_info and t_info.get('evaluated') and t_info.get('alt_gain', 0) <= 0:
                                        thermal_already_evaluated_bad = True
                                
                                if thermal_already_evaluated_bad:
                                    print(f"⚠️  UAV {u}: Thermique de secours {target_tid} atteinte mais déjà évaluée non viable → Ignorée")
                                    FLT_track[u]['seeking_thermal'] = False
                                    FLT_track[u]['seeking_thermal_id'] = None
                                    FLT_track[u]['seeking_wp_idx'] = None
                                    FLT_track[u]['visited_thermals'].add(target_tid)
                                else:
                                    print(f"✅ UAV {u}: Thermique de secours {target_tid} atteinte → Lancement orbite")
                                    FLT_track[u]['seeking_thermal'] = False
                                    FLT_track[u]['seeking_thermal_id'] = None
                                    FLT_track[u]['seeking_wp_idx'] = None
                                    
                                    # Vérifier si déjà évaluée comme viable → soaring direct
                                    t_info_rescue = thermal_map.get_info(target_tid) if thermal_map else None
                                    if t_info_rescue and t_info_rescue.get('evaluated') and t_info_rescue.get('alt_gain', 0) > 0:
                                        # Thermique connue viable → orbite soaring directe
                                        alt_conflict = _check_thermal_altitude_conflict(
                                            u, target_tid, FLT_track[u]['Z'][-1],
                                            controller, FLT_track, nUAVs
                                        )
                                        if not alt_conflict:
                                            FLT_track[u]['current_thermal_id'] = target_tid
                                            FLT_track[u]['flight_mode'].append('soaring')
                                            FLT_track[u]['soaring_start_time'] = current_time
                                            print(f"  UAV {u}: Thermique {target_tid} connue viable → Soaring MAVSDK")
                                            orbit_ok = await controller.start_thermal_orbit(u, t_obj, FLT_track[u]['Z'][-1], mode='soaring')
                                            if not orbit_ok:
                                                FLT_track[u]['flight_mode'][-1] = 'glide'
                                                FLT_track[u]['soaring_start_time'] = None
                                                FLT_track[u]['current_thermal_id'] = None
                                                print(f"  UAV {u}: Échec orbite soaring secours")
                                            else:
                                                FLT_track[u]['thermal_exploitation_log'].append({
                                                    'thermal_id': target_tid,
                                                    'entry_time': current_time,
                                                    'exit_time': None,
                                                })
                                        else:
                                            print(f"  UAV {u}: Thermique {target_tid} occupée à altitude similaire")
                                    else:
                                        # Thermique inconnue → orbite d'évaluation
                                        if thermal_map:
                                            thermal_map.ensure_exists(target_tid, t_obj, current_time)
                                        FLT_track[u]['in_evaluation'] = True
                                        FLT_track[u]['current_thermal_id'] = target_tid
                                        FLT_track[u]['evaluation_start_altitude'] = FLT_track[u]['Z'][-1]
                                        
                                        evaluation_obstacle = {
                                            'X': t_obj.x, 'Y': t_obj.y,
                                            'radius': t_obj.radius,
                                            'type': 'evaluation_zone', 'uav_id': u
                                        }
                                        eval_poly = convert_cylindrical_obstacles_to_polygons([evaluation_obstacle])
                                        params['obstacles'].append({
                                            'vertices': eval_poly[0], 'uav_id': u,
                                            'thermal_id': target_tid, 'type': 'evaluation_zone'
                                        })
                                        
                                        print(f"  UAV {u}: Thermique {target_tid} inconnue → Évaluation MAVSDK")
                                        orbit_ok = await controller.start_thermal_orbit(u, t_obj, FLT_track[u]['Z'][-1], mode='evaluation')
                                        if not orbit_ok:
                                            FLT_track[u]['in_evaluation'] = False
                                            FLT_track[u]['current_thermal_id'] = None
                                            _remove_evaluation_obstacles(params, u, target_tid)
                                            print(f"  UAV {u}: Échec orbite évaluation secours")
                        
                        # Si seeking est toujours actif (pas annulé), skip le check altitude
                        if not seeking_cancelled:
                            continue
                        # Sinon, tomber dans le check altitude ci-dessous
                    
                    current_alt = FLT_track[u]['Z'][-1]
                    # Endurance : chercher proactivement dès que sous le working_floor
                    # Autres scénarios : seulement en secours à basse altitude
                    seeking_threshold = params['working_floor'] if endurance_seeking else (z_lower + low_alt_margin)
                    if current_alt <= seeking_threshold:
                        current_x = FLT_track[u]['X'][-1]
                        current_y = FLT_track[u]['Y'][-1]
                        current_airspeed = FLT_conditions[u].get('airspeed', 13.0)
                        
                        sink_rate = abs(get_sink_rate(UAV_data, FLT_conditions[u]))
                        glide_ratio = current_airspeed / sink_rate if sink_rate > 0 else 10.0
                        altitude_available = current_alt - z_lower
                        max_glide_range = altitude_available * glide_ratio * 0.8
                        # Endurance : portée plus large (pas de marge de sécurité 0.8)
                        if endurance_seeking:
                            max_glide_range = altitude_available * glide_ratio
                        
                        best_thermal_id = None
                        best_distance = float('inf')
                        
                        # Endurance : considérer toutes les thermiques connues (ROS bridge)
                        # Autres scénarios : seulement les thermiques physiquement détectées
                        if endurance_seeking:
                            candidate_thermals = active_thermals or {}
                        else:
                            candidate_thermals = thermal_map.get_detected_thermals() if thermal_map else {}
                        for tid, t_obj in candidate_thermals.items():
                            thermal = t_obj
                            if not thermal.is_active(current_time):
                                continue
                            # Ignorer les thermiques déjà visitées (sauf si altitude < working_floor)
                            if tid in FLT_track[u]['visited_thermals']:
                                continue
                            # Ignorer les thermiques déjà évaluées comme non viables
                            if thermal_map:
                                t_info = thermal_map.get_info(tid)
                                if t_info and t_info.get('evaluated') and t_info.get('alt_gain', 0) <= 0:
                                    continue
                            dist = np.sqrt((current_x - thermal.x)**2 + (current_y - thermal.y)**2)
                            if dist < max_glide_range and dist < best_distance:
                                # Vérifier séparation altitude avec drones déjà dans cette thermique
                                alt_conflict = _check_thermal_altitude_conflict(
                                    u, tid, current_alt,
                                    controller, FLT_track, nUAVs
                                )
                                if not alt_conflict:
                                    best_thermal_id = tid
                                    best_distance = dist
                        
                        if best_thermal_id is not None:
                            thermal = active_thermals[best_thermal_id]
                            wp_idx = current_wp_indices[u]
                            time_to_reach = best_distance / current_airspeed if current_airspeed > 0 else 999
                            alt_at_arrival = current_alt - (sink_rate * time_to_reach)
                            
                            GOAL_WPs[u]['X'].insert(wp_idx, thermal.x)
                            GOAL_WPs[u]['Y'].insert(wp_idx, thermal.y)
                            GOAL_WPs[u]['Z'].insert(wp_idx, max(alt_at_arrival, z_lower))
                            
                            FLT_track[u]['seeking_thermal'] = True
                            FLT_track[u]['seeking_thermal_id'] = best_thermal_id
                            FLT_track[u]['seeking_wp_idx'] = wp_idx
                            
                            if endurance_seeking:
                                print(f"\n🔋 UAV {u}: Recherche proactive thermique ({current_alt:.0f}m < floor {params['working_floor']:.0f}m) → "
                                      f"Thermique {best_thermal_id} "
                                      f"(dist={best_distance:.0f}m, arrivée≈{alt_at_arrival:.0f}m)")
                            else:
                                print(f"\n⚠️  UAV {u}: Altitude basse ({current_alt:.0f}m) → "
                                      f"Redirection vers thermique {best_thermal_id} "
                                      f"(dist={best_distance:.0f}m, arrivée≈{alt_at_arrival:.0f}m, "
                                      f"portée max={max_glide_range:.0f}m)")
            
            # ========== FIN GESTION UAV ==========
            
            # Attendre pour synchroniser avec le temps réel si le calcul est trop rapide
            time_to_wait = params['target_real_time_per_iteration'] - algo_execution_time
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
            
            # Détection d'objets pour scénario couverture
            if scenario == TestScenario.COVERAGE and surveillance_objects:
                for obj in surveillance_objects:
                    if not obj.is_active(current_time) or obj.detected:
                        continue
                    for u in range(nUAVs):
                        if len(FLT_track[u]['X']) > 0 and obj.is_in_fov(
                            FLT_track[u]['X'][-1], FLT_track[u]['Y'][-1],
                            FLT_track[u]['Z'][-1], fov_radius
                        ):
                            obj.detected = True
                            if u not in obj.detected_by:
                                obj.detected_by.append(u)
            
            iteration += 1
            
            # Mise à jour PX4 pour tous les UAVs
            await controller.update_all_from_simulation(FLT_track, FLT_conditions)
            
            # Vérifier périodiquement si un drone a perdu le mode offboard
            # (toutes les 10 itérations ≈ 11s pour éviter du spam)
            if iteration % 10 == 0:
                await controller.recover_offboard_all(FLT_track)
            
            # Verification atterissage d'urgence (batterie faible)
            for u in range(nUAVs):
                bridge = controller.bridges[u]
                if bridge.is_landing:
                    continue
                if FLT_track[u]['battery_capacity'][-1] <= UAV_data['desired_reserved_battery_capacity']:
                    print(f"\n⚠️  UAV {u} batterie faible ({FLT_track[u]['battery_capacity'][-1]:.1f}Ah) - Atterrissage d'urgence!")
                    # Marquer comme en atterrissage dans FLT_track (pour gotoWaypointMulti)
                    FLT_track[u]['flight_mode'].append('landing')
                    # Arrêter l'orbite si en cours avant l'atterrissage
                    if bridge.is_orbiting:
                        await controller.stop_thermal_orbit(u)
                    # Lancer l'atterrissage en tâche de fond (ne bloque pas la boucle)
                    asyncio.create_task(bridge.return_and_land())
                    
            
            # Affichage périodique
            if iteration % 5 == 0:
                current_real_time = time.perf_counter()
                total_real_time = current_real_time - real_time_start
                avg_algo_time = (total_decision_time / decision_calls) * 1000
                
                # Calculer le décalage de synchronisation
                ideal_real_time = params['current_simulation_time']  # Idéalement 1:1
                sync_ratio = params['current_simulation_time'] / total_real_time if total_real_time > 0 else 0
                sync_delay = total_real_time - params['current_simulation_time']
                
                # Qualité de synchronisation
                if 0.95 <= sync_ratio <= 1.05:
                    sync_status = "✅ EXCELLENT"
                elif 0.8 <= sync_ratio <= 1.2:
                    sync_status = "✓ BON"
                elif 0.5 <= sync_ratio <= 2.0:
                    sync_status = "⚠️  MOYEN"
                else:
                    sync_status = "❌ MAUVAIS"
                
                print(f"\n{'='*70}")
                print(f"[Simulation: {current_time:.1f}s | Réel: {total_real_time:.1f}s | Iter: {iteration}]")
                print(f"  Temps calcul: {avg_algo_time:.1f}ms (objectif: {params['target_real_time_per_iteration']*1000:.0f}ms)")
                print(f"  Sync temps réel: {sync_ratio:.2f}x {sync_status}")
                print(f"  Décalage: {sync_delay:+.2f}s | Steps G: B={params['bearing_step_glide']}, V={params['speed_step_glide']} | E: B={params['bearing_step_engine']}, V={params['speed_step_engine']}")
                print(f"{'='*70}")
                
                for u in range(nUAVs):
                    if controller.bridges[u].is_landing:
                        print(f"  UAV {u}: ✈️  ATTERRI / EN ATTERRISSAGE")
                        continue
                    if len(FLT_track[u]['X']) > 0:
                        # Vérifier les NaN
                        x = FLT_track[u]['X'][-1]
                        y = FLT_track[u]['Y'][-1]
                        z = FLT_track[u]['Z'][-1]
                        
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            print(f"  UAV {u}: ⚠️  ERREUR - Position invalide (NaN)")
                            continue
                        
                        pos_str = f"({x:.0f}, {y:.0f}, {z:.0f})"
                        mode = FLT_track[u]['flight_mode'][-1]
                        battery = FLT_track[u]['battery_capacity'][-1]
                        airspeed = FLT_conditions[u]['airspeed']
                        
                        status = f"UAV {u}: {pos_str} | {mode} | V:{airspeed:.1f}m/s | Bat: {battery:.1f}"
                        
                        # Ajouter info thermique si applicable
                        if FLT_track[u]['current_thermal_id'] is not None:
                            status += f" | Therm: {FLT_track[u]['current_thermal_id']}"
                        
                        # Ajouter info orbite MAVSDK si applicable
                        if controller.bridges[u].is_orbiting:
                            alt_var = controller.bridges[u].get_altitude_variation()
                            status += f" | ORBIT({controller.bridges[u].orbit_mode})"
                            status += f" Δalt={alt_var['total_change']:+.1f}m"
                            status += f" climb={alt_var['avg_climb_rate']:.2f}m/s"
                        
                        print(f"  {status}")
        
        # ========== FIN DE SIMULATION ==========
        # Arrêter toutes les orbites en cours avant l'atterrissage
        for u in range(nUAVs):
            if controller.bridges[u].is_landing:
                continue
            if controller.bridges[u].is_orbiting:
                print(f"  Arrêt orbite UAV {u}...")
                await controller.stop_thermal_orbit(u)
        
        total_real_time_final = time.perf_counter() - real_time_start
        
        print("\n" + "="*70)
        print("FIN DE LA SIMULATION MULTI-UAV")
        print("="*70)
        print(f"Temps simulé: {params['current_simulation_time']:.1f}s")
        print(f"Temps réel: {total_real_time_final:.1f}s")
        print(f"Ratio: {params['current_simulation_time']/total_real_time_final:.2f}x")
        print(f"Itérations: {iteration}")
        print(f"Time step moyen: {params['current_simulation_time']/iteration:.2f}s")
        
        # ========== ANALYSE DES PERFORMANCES ==========
        analyzer = PerformanceAnalyzer()
        
        # VALIDATION : Vérifier l'intégrité des données de vol avant analyse
        print("\n🔍 Validation des données de vol...")
        data_valid = True
        for u in range(nUAVs):
            x_len = len(FLT_track[u]['X'])
            y_len = len(FLT_track[u]['Y'])
            z_len = len(FLT_track[u]['Z'])
            time_len = len(FLT_track[u]['flight_time'])
            mode_len = len(FLT_track[u]['flight_mode'])
            
            if not (x_len == y_len == z_len == time_len == mode_len):
                print(f"⚠️  UAV {u}: Longueurs incohérentes - X:{x_len}, Y:{y_len}, Z:{z_len}, time:{time_len}, mode:{mode_len}")
                data_valid = False
            
            # Si flight_time est vide ou incomplet, le reconstruire
            if time_len == 0 or time_len < x_len:
                print(f"⚠️  UAV {u}: flight_time incomplet ({time_len}/{x_len}) - reconstruction...")
                FLT_track[u]['flight_time'] = [i * params['time_step'] for i in range(x_len)]
        
        if data_valid:
            print("✓ Données de vol validées")

        # Fermer les exploitations thermiques restées ouvertes en fin de simulation
        for u in range(nUAVs):
            for entry in FLT_track[u].get('thermal_exploitation_log', []):
                if entry.get('exit_time') is None:
                    entry['exit_time'] = params['current_simulation_time']
        
        # Calculer les métriques de trajectoire
        try:
            path_metrics = analyzer.calculate_path_metrics(FLT_track, nUAVs)
            for u in range(nUAVs):
                metrics.total_distance[u] = path_metrics[u]['distance']
                metrics.path_length[u] = path_metrics[u]['waypoints']
        except Exception as e:
            print(f"⚠️  Erreur calcul métriques trajectoire: {e}")
        
        # Calculer les métriques de temps
        try:
            phase_times = analyzer.calculate_flight_phase_times(FLT_track, nUAVs)
            for u in range(nUAVs):
                metrics.total_flight_time[u] = phase_times[u]['total']
                metrics.glide_time[u] = phase_times[u]['glide']
                metrics.soar_time[u] = phase_times[u]['soar']
                metrics.engine_time[u] = phase_times[u]['engine']
                total = phase_times[u]['total']
                motor_off = phase_times[u]['glide'] + phase_times[u]['soar']
                metrics.motor_off_ratio[u] = (motor_off / total * 100.0) if total > 0 else 0.0
        except Exception as e:
            print(f"⚠️  Erreur calcul métriques temps: {e}")
        
        # Calculer la distance minimale de séparation
        metrics.min_separation_distance = analyzer.check_min_separation(FLT_track, nUAVs)
        metrics.collisions_avoided = sum(FLT_track[u].get('collisions_avoided_count', 0) for u in range(nUAVs))

        # M11: temps de calcul par itération
        if decision_calls > 0:
            metrics.algorithm_time_avg_ms = (total_decision_time / decision_calls) * 1000.0
            metrics.algorithm_time_max_ms = max_decision_time * 1000.0
        
        # Calculer les métriques de batterie
        for u in range(nUAVs):
            if len(FLT_track[u]['battery_capacity']) > 0:
                metrics.battery_remaining[u] = FLT_track[u]['battery_capacity'][-1]
                metrics.battery_consumed[u] = UAV_data['maximum_battery_capacity'] - metrics.battery_remaining[u]
                total = metrics.total_flight_time.get(u, 0.0)
                metrics.battery_consumption_rate_ah_per_h[u] = (
                    metrics.battery_consumed[u] / (total / 3600.0)
                    if total > 0 else 0.0
                )

        # M3: variation d'énergie potentielle
        metrics.potential_energy_j = analyzer.calculate_potential_energy_variation(
            FLT_track,
            nUAVs,
            uav_mass=UAV_data.get('weight', 1.6),
            g=grav_accel,
        )

        # M4: thermiques détectées par UAV
        for u in range(nUAVs):
            metrics.thermals_detected_per_uav[u] = FLT_track[u].get('thermals_detected_count', 0)

        # M6/M7: fréquence et durées d'exploitation par thermique depuis les logs
        thermal_log_metrics = analyzer.calculate_thermal_exploitation_log_metrics(FLT_track, nUAVs)
        metrics.thermal_exploitation_frequency = thermal_log_metrics['frequency']
        metrics.thermal_exploitation_duration_s = thermal_log_metrics['duration_per_thermal_s']
        metrics.thermal_exploitation_duration_per_uav_s = thermal_log_metrics['duration_per_uav_s']
        metrics.thermal_exploited_unique = len(metrics.thermal_exploitation_frequency)
        
        # Analyser les thermiques pour le scénario d'endurance
        if scenario == TestScenario.ENDURANCE and len(active_thermals) > 0:
            # Convertir le dictionnaire de thermiques en liste pour l'analyse
            thermals_list = list(active_thermals.values()) if isinstance(active_thermals, dict) else active_thermals
            thermal_analysis = analyzer.analyze_thermal_exploitation(thermals_list, FLT_track, nUAVs)
            metrics.thermals_detected = thermal_analysis['detected']
            metrics.thermals_exploited = thermal_analysis['exploited']
            metrics.thermals_rejected = thermal_analysis['rejected']
            metrics.thermals_per_uav = thermal_analysis['per_uav']
            
            # Métriques d'endurance : boucles de patrouille et ratio soaring
            metrics.patrol_loops = patrol_loops
            for u in range(nUAVs):
                total = metrics.total_flight_time.get(u, 0)
                soar = metrics.soar_time.get(u, 0)
                metrics.soaring_ratio[u] = soar / total if total > 0 else 0.0

        # M5: ratio exploitées/détectées par UAV (+ agrégat)
        total_detected = 0
        total_exploited = 0
        for u in range(nUAVs):
            detected_u = metrics.thermals_detected_per_uav.get(u, 0)
            exploited_u = metrics.thermals_per_uav.get(u, 0)
            metrics.exploited_detected_ratio[u] = (
                exploited_u / detected_u * 100.0 if detected_u > 0 else 0.0
            )
            total_detected += detected_u
            total_exploited += exploited_u
        metrics.thermals_detected = total_detected
        if total_exploited > metrics.thermals_exploited:
            metrics.thermals_exploited = total_exploited

        # M5+: taux global d'exploitation thermique (N_exp unique / N_total)
        if metrics.thermals_generated > 0:
            metrics.thermal_exploitation_global_ratio = (
                metrics.thermal_exploited_unique / metrics.thermals_generated * 100.0
            )
        else:
            metrics.thermal_exploitation_global_ratio = 0.0
        
        # Analyser la couverture pour le scénario de couverture
        if scenario == TestScenario.COVERAGE and surveillance_objects:
            coverage_analysis = analyzer.analyze_coverage(surveillance_objects, FLT_track, nUAVs, fov_radius)
            metrics.objects_detected = coverage_analysis['detected']
            metrics.detection_rate = coverage_analysis['detection_rate']
            metrics.objects_per_uav = coverage_analysis['per_uav']
        
        # Afficher le rapport de performance
        analyzer.print_performance_report(metrics)
        
        # Sauvegarder les métriques
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_json = f"metrics_{scenario.value}_{nUAVs}uavs_{timestamp}.json"
        analyzer.save_metrics_to_file(metrics, filename_json)
        filename_csv = f"metrics_{scenario.value}_{nUAVs}uavs_{timestamp}.csv"
        analyzer.save_metrics_to_csv(metrics, filename_csv)
        
        # Statistiques par UAV
        print("\nStatistiques par UAV:")
        for u in range(nUAVs):
            if len(FLT_track[u]['X']) > 0:
                final_pos = f"({FLT_track[u]['X'][-1]:.0f}, {FLT_track[u]['Y'][-1]:.0f}, {FLT_track[u]['Z'][-1]:.0f})"
                battery = FLT_track[u]['battery_capacity'][-1]
                flight_time = FLT_track[u]['flight_time'][-1]
                
                # Compter les modes
                mode_counts = {}
                for mode in FLT_track[u]['flight_mode']:
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
                
                print(f"\n  UAV {u}:")
                print(f"    Position finale: {final_pos}")
                print(f"    Batterie: {battery:.2f}/{UAV_data['maximum_battery_capacity']:.2f}")
                print(f"    Temps de vol: {flight_time:.1f}s")
                print(f"    Modes: {mode_counts}")
        
        # Atterrir tous les UAVs
        await controller.land_all()
        
        # Arrêter le bridge ROS2
        thermal_bridge.stop()
        
        print("\n✓ Mission multi-UAV terminée avec succès!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption - Atterrissage d'urgence de tous les UAVs")
        await controller.land_all()
        thermal_bridge.stop()
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        try:
            await controller.land_all()
            thermal_bridge.stop()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(run_multi_uav_simulation())