"""
Integration multi-UAV de votre simulation de planeur avec PX4 SITL
Support de plusieurs drones avec thermiques partagées
"""

import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw, AttitudeRate
from mavsdk.telemetry import LandedState
import time
import pymap3d as pm

# Importer vos modules existants
from GoToWP import gotoWaypointMulti
from trajectory import TrajectoryEvaluator, generate_all_trajectories, generate_random_obstacles, LawnMowerTrajectory
from thermal import ThermalGenerator, ThermalMap, ThermalEvaluator
from Scenario import (TestScenario, PerformanceMetrics, SurveillanceObject, 
                      ScenarioGenerator, PerformanceAnalyzer, select_scenario)


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
        
        # Taux de mise à jour (Hz)
        self.update_rate = 10
        
        # Keepalive pour offboard
        self.offboard_keepalive_task = None
        self.last_position_ned = PositionNedYaw(0.0, 0.0, 0.0, 0.0)
        self.last_velocity_ned = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    
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
        
        # Stocker la position home
        async for terrain_info in self.drone.telemetry.home():
            self.home_position = terrain_info
            print(f"[UAV {self.uav_id}] ✓ Position home: {terrain_info.latitude_deg:.6f}, {terrain_info.longitude_deg:.6f}")
            break
        
        # Définir l'origine de la simulation
        self.simulation_origin = {
            'lat': self.home_position.latitude_deg,
            'lon': self.home_position.longitude_deg,
            'alt': self.home_position.absolute_altitude_m
        }
    
    async def arm_and_takeoff(self):
        """Armement et décollage"""
        
        await self.drone.action.hold()
        
        print(f"[UAV {self.uav_id}] Armement...")
        await self.drone.action.arm()
        
        print(f"[UAV {self.uav_id}] Décollage...")
        await self.drone.action.takeoff()
        
        target_alt = 400
        async for position in self.drone.telemetry.position():
            current_alt = position.relative_altitude_m
            print(f"[UAV {self.uav_id}] Altitude actuelle: {current_alt:.1f}m / {target_alt:.1f}m", end='\r')
            if current_alt >= target_alt:
                print(f"[UAV {self.uav_id}] ✓ Altitude atteinte: {current_alt:.1f}m")
                break
    
    async def _offboard_keepalive_loop(self):
        """
        Boucle keepalive pour maintenir le mode offboard actif
        Envoie des setpoints à 10Hz minimum
        """
        print(f"[UAV {self.uav_id}] Démarrage keepalive offboard...")
        try:
            while True:
                await self.drone.offboard.set_position_velocity_ned(
                    self.last_position_ned,
                    self.last_velocity_ned
                )
                await asyncio.sleep(0.05)  # 20Hz pour la sécurité (minimum requis: 2Hz)
        except asyncio.CancelledError:
            print(f"[UAV {self.uav_id}] Arrêt keepalive offboard")
        except Exception as e:
            print(f"[UAV {self.uav_id}] Erreur keepalive: {e}")
    
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
        
        # Origine de référence
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
        
        # Convertir ECEF en NED par rapport à l'origine
        north_ned, east_ned, down_ned = pm.ecef2ned(
            x_ecef, y_ecef, z_ecef,
            lat0, lon0, h0
        )
        
        # Orientation (bearing en radians)
        bearing_rad = FLT_track['bearing'][-1]
        
        # VALIDATION : Vérifier le bearing
        if np.isnan(bearing_rad) or np.isinf(bearing_rad):
            print(f"[UAV {self.uav_id}] ❌ Bearing invalide: {bearing_rad}")
            return
        
        yaw_deg = np.degrees(bearing_rad)
        # Normaliser le yaw dans [0, 360[
        yaw_deg = yaw_deg % 360.0
        
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
        # Vitesse horizontale
        v_horizontal = airspeed * np.cos(flight_path_angle)
        
        # Composantes Est et Nord basées sur le bearing
        velocity_east = v_horizontal * np.sin(bearing_rad)
        velocity_north = v_horizontal * np.cos(bearing_rad)
        velocity_up = airspeed * np.sin(flight_path_angle)
        
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
        self.last_position_ned = PositionNedYaw(north_ned, east_ned, down_ned, yaw_deg)
        self.last_velocity_ned = VelocityNedYaw(velocity_north_ned, velocity_east_ned, velocity_down_ned, yaw_deg)
        
        # La commande est maintenant envoyée par la boucle keepalive
        # pas besoin d'envoyer ici pour éviter les conflits
    
    async def start_offboard_mode(self):
        """Démarrer le mode offboard avec contrôle en vitesse"""
        print(f"[UAV {self.uav_id}] Démarrage mode offboard ...")
        await self.drone.offboard.set_position_ned(
               PositionNedYaw(0.0, 0.0, 0.0, 0.0)
            )
        
        try:
            await self.drone.offboard.start()
            print(f"[UAV {self.uav_id}] ✓ Mode offboard activé ")
            
            # Démarrer la boucle keepalive
            self.offboard_keepalive_task = asyncio.create_task(self._offboard_keepalive_loop())
            
            return True
        except OffboardError as error:
            print(f"[UAV {self.uav_id}] ✗ Erreur offboard: {error}")
            return False
    
    async def stop_offboard_mode(self):
        """Arrêter le mode offboard"""
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
        """Retour et atterrissage"""
        print(f"[UAV {self.uav_id}] Retour à la base...")
        await self.stop_offboard_mode()
        await self.drone.action.return_to_launch()
        
        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                print(f"[UAV {self.uav_id}] ✓ Atterri")
                break
            await asyncio.sleep(1)


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
        """Démarrer le mode offboard pour tous les UAVs"""
        print("\nDémarrage du mode offboard pour tous les UAVs...")
        tasks = []
        for u in range(self.nUAVs):
            tasks.append(asyncio.create_task(self.bridges[u].start_offboard_mode()))
        results = await asyncio.gather(*tasks)
        
        success_count = sum(results)
        if success_count == self.nUAVs:
            print(f"✓ Mode offboard activé pour tous les {self.nUAVs} UAVs")
            return True
        else:
            print(f"⚠️  Mode offboard activé pour {success_count}/{self.nUAVs} UAVs")
            return False
    
    async def update_all_from_simulation(self, FLT_track, FLT_conditions):
        """Mettre à jour tous les UAVs avec l'état de simulation"""
        update_tasks = []
        for u in range(self.nUAVs):
            if u < len(self.bridges):
                update_tasks.append(
                    asyncio.create_task(self.bridges[u].update_from_simulation_state(FLT_track[u], FLT_conditions[u]))
                )
        
        await asyncio.gather(*update_tasks)
    
    async def land_all(self):
        """Atterrir tous les UAVs"""
        print("\n" + "="*70)
        print("ATTERRISSAGE DE TOUS LES UAVs")
        print("="*70)
        tasks = []
        for u in range(self.nUAVs):
            tasks.append(asyncio.create_task(self.bridges[u].return_and_land()))
        await asyncio.gather(*tasks)
        
        print("\n✓ Tous les UAVs ont atterri!")
        
    async def set_altitude_all(self, target_altitude):
        """Définir l'altitude cible pour tous les UAVs"""
        print(f"\nDéfinition de l'altitude cible à {target_altitude}m pour tous les UAVs...")
        tasks = []
        for u in range(self.nUAVs):
            tasks.append(asyncio.create_task(self.bridges[u].drone.action.do_orbit(
                100,15,3,self.bridges[u].home_position.latitude_deg, self.bridges[u].home_position.longitude_deg, target_altitude
            )))
        await asyncio.gather(*tasks)
        print(f"\n✓ Altitude cible définie à {target_altitude}m pour tous les UAVs!")


def generate_lawnmower_trajectories(nUAVs, params, UAV_data, fov_radius):
    """
    Génère des trajectoires LawnMower pour tous les UAVs pour une couverture maximale
    
    Args:
        nUAVs (int): Nombre de drones
        params (dict): Paramètres de simulation
        UAV_data (dict): Données des UAVs
        fov_radius (float): Rayon du champ de vision
        coverage_area (dict): Zone à couvrir {X_min, X_max, Y_min, Y_max} (optionnel)
        
    Returns:
        dict: Trajectoires pour chaque UAV {0: {X: [], Y: [], Z: []}, ...}
    """
    
    x_min = params['X_lower_bound']
    x_max = params['X_upper_bound']
    y_min = params['Y_lower_bound']
    y_max = params['Y_upper_bound']
    altitude = param['working_floor']
    
    coverage_area = {
        'X_min': x_min,
        'X_max': x_max,
        'Y_min': y_min,
        'Y_max': y_max,
    }
    
    # Créer le générateur de trajectoire LawnMower
    lawnmower = LawnMowerTrajectory(params, UAV_data)
    
    # Générer les trajectoires pour chaque UAV
    trajectories = {}
    for u in range(nUAVs):
        trajectory = lawnmower.generate_path(
            area_bounds=coverage_area,
            fov_radius=fov_radius,
            uav_id=u,
            num_uavs=nUAVs,
            altitude=altitude
        )
        trajectories[u] = trajectory
        
        print(f"  UAV {u}: {len(trajectory['X'])} waypoints générés")
    
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
    params['X_lower_bound'] = 0.0
    params['X_upper_bound'] = 6000.0
    params['Y_lower_bound'] = 0.0
    params['Y_upper_bound'] = 6000.0
    params['Z_lower_bound'] = 200.0
    params['Z_upper_bound'] = 1000.0
    params['current_simulation_time'] = 0.0
    params['time_step'] = 1.1  # Time step cible pour synchronisation temps réel (secondes)
    params['target_real_time_per_iteration'] = 1.1  # Temps réel cible par itération
    params['bearing_step'] = 4  # Réduit de 10 à 5 pour accélérer (50% moins de calculs)
    params['speed_step'] = 3  # Réduit de 10 à 3 pour accélérer (70% moins de calculs)
    params['safe_distance'] = 30.0
    params['horizon_length'] = 100.0
    params['adaptive_resolution'] = True  # Ajuster automatiquement la résolution si trop lent
    
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
        
        # Calculer le centroïde
        center_x = 3000.0
        center_y = 3000.0
        center_z = 400.0  # 400m au-dessus
        
        # ========== GÉNÉRATION SCÉNARIO BASÉE SUR LES POSITIONS HOME ==========
        scenario_gen = ScenarioGenerator()
        surveillance_objects = None
        active_thermals = dict()
        thermal_map = ThermalMap()
        thermal_evaluator = ThermalEvaluator(params, UAV_data)
        
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
            thermal_generator = ThermalGenerator(params)
            num_thermals = max(2, nUAVs // 3)
            active_thermals = thermal_generator.generate_random_thermals(
                num_thermals, obstacles, params['current_simulation_time']
            )
            print(f'✓ {len(active_thermals)} thermiques actives')
            print(f"\n✓ Scénario trajectoire optimale (planeur) vers ({end_position['X']:.0f}, {end_position['Y']:.0f})")
            
        elif scenario == TestScenario.ENDURANCE:
            # Positions de départ = positions home réelles des UAVs
            thermal_generator = ThermalGenerator(params)
            obstacles = generate_random_obstacles(3, params)
            params['obstacles'] = obstacles
            active_thermals, thermal_stats = scenario_gen.generate_endurance_scenario(nUAVs, params, thermal_generator, obstacles)
            metrics.thermals_generated = len(active_thermals)
            allow_glide = True
            # Utiliser les positions home réelles avec bearing vers le centre
            start_positions = {}
            for u in range(nUAVs):
                # Calculer le bearing vers le centre depuis la position home
                dx = center_x - home_positions[u]['X']
                dy = center_y - home_positions[u]['Y']
                bearing_to_center = np.arctan2(dx, dy)
                
                start_positions[u] = {
                    'X': home_positions[u]['X'],
                    'Y': home_positions[u]['Y'],
                    'Z': home_positions[u]['Z'] + 400.0,  # 400m au-dessus de home
                    'bearing': bearing_to_center
                }
            end_position = {'X': center_x, 'Y': center_y, 'Z': center_z}
            print(f"\n✓ Scénario endurance: {len(active_thermals)} thermiques, départ des positions home")
        
        elif scenario == TestScenario.COVERAGE:
            mission_duration = 1200  # 20 minutes
            surveillance_objects = scenario_gen.generate_coverage_scenario(nUAVs, params, mission_duration)
            metrics.objects_total = len(surveillance_objects)
            obstacles = generate_random_obstacles(5, params)
            params['obstacles'] = obstacles
            allow_glide = True
            thermal_generator = ThermalGenerator(params)
            num_thermals = max(3, nUAVs // 2)
            active_thermals = thermal_generator.generate_random_thermals(
                num_thermals, obstacles, params['current_simulation_time']
            )
            print(f'✓ {len(active_thermals)} thermiques actives')
            
            print(f"\n✓ Scénario couverture: {len(surveillance_objects)} objets à surveiller")
        
        
        # ========== CALCULS ATMOSPHÉRIQUES ==========
        ACC_SEA_LEVEL = 9.80665
        T_SEA_LEVEL = 288.15
        RHO_SEA_LEVEL = 1.225
        MEAN_EARTH_RADIUS = 6371009
        TROPO_LAPSE_RATE = -0.0065
        R = 287.058
        
        grav_accel = ACC_SEA_LEVEL * (MEAN_EARTH_RADIUS / (MEAN_EARTH_RADIUS + params['working_floor']))
        T_fin = T_SEA_LEVEL + TROPO_LAPSE_RATE * params['working_floor']
        air_density = RHO_SEA_LEVEL * (T_fin / T_SEA_LEVEL)**(-grav_accel / (TROPO_LAPSE_RATE * R) - 1)
        
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

            END_WPs[u]['X'].append(end_position['X'])
            END_WPs[u]['Y'].append(end_position['Y'])
            END_WPs[u]['Z'].append(end_position['Z'])

            # Utiliser les positions de départ du scénario (basées sur home positions)
            initial_x = start_positions[u]['X']
            initial_y = start_positions[u]['Y']
            initial_z = start_positions[u]['Z']
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
        max_iterations = 3000 if scenario == TestScenario.COVERAGE else 2000
        
        total_decision_time = 0
        decision_calls = 0
        
        # Horloge temps réel (pour statistiques seulement)
        real_time_start = time.perf_counter()
        
        print(f"\n⚙️  Configuration temps réel:")
        print(f"   Time step cible: {params['time_step']:.2f}s")
        print(f"   Bearing steps: {params['bearing_step']} (optimisé pour temps réel)")
        print(f"   Speed steps: {params['speed_step']} (optimisé pour temps réel)")
        print(f"   Résolution adaptive: {'Activée' if params['adaptive_resolution'] else 'Désactivée'}")
        print("   → Objectif: calcul en ~{:.0f}ms pour réactivité maximale\n".format(params['target_real_time_per_iteration']*1000))
        
        while iteration < max_iterations:
            # Vérifier si tous les UAVs ont terminé
            if all(current_wp_indices[u] >= len(GOAL_WPs[u]['X']) for u in range(nUAVs)):
                print("\n✓ Tous les UAVs ont atteint leurs objectifs!")
                break
            
            # Mesurer le temps de décision de l'algorithme
            algo_start_time = time.perf_counter()
            
            # Mise à jour de la simulation
            if thermal_map and thermal_evaluator:
                FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices, SOAR_WPs, current_soar_wp_indices = gotoWaypointMulti(
                    FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data,
                    current_wp_indices, current_eval_wp_indices, current_soar_wp_indices, 
                    thermal_map, thermal_evaluator, EVAL_WPs, active_thermals, SOAR_WPs
                )
            else:
                # Version simplifiée sans thermiques
                FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices, SOAR_WPs, current_soar_wp_indices = gotoWaypointMulti(
                    FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data,
                    current_wp_indices, current_eval_wp_indices, current_soar_wp_indices, 
                    None, None, EVAL_WPs, [], SOAR_WPs
                )
            
            algo_end_time = time.perf_counter()
            algo_execution_time = algo_end_time - algo_start_time
            
            total_decision_time += algo_execution_time
            decision_calls += 1
            
            # SYNCHRONISATION TEMPS RÉEL INTELLIGENTE
            # Si le calcul est trop lent, réduire la résolution adaptativement
            if params['adaptive_resolution'] and algo_execution_time > params['target_real_time_per_iteration'] * 2:
                if params['bearing_step'] > 3:
                    params['bearing_step'] = max(3, params['bearing_step'] - 1)
                    print(f"⚠️  Calcul trop lent ({algo_execution_time:.2f}s) - Réduction bearing_step à {params['bearing_step']}")
                if params['speed_step'] > 2:
                    params['speed_step'] = max(2, params['speed_step'] - 1)
                    print(f"⚠️  Calcul trop lent - Réduction speed_step à {params['speed_step']}")
            
            # Mettre à jour le temps de simulation
            params['current_simulation_time'] += params['time_step']
            current_time = params['current_simulation_time']
            
            # Attendre pour synchroniser avec le temps réel si le calcul est trop rapide
            time_to_wait = params['target_real_time_per_iteration'] - algo_execution_time
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
            
            # Détection d'objets pour scénario couverture
            if scenario == TestScenario.COVERAGE and surveillance_objects:
                for obj in surveillance_objects:
                    if obj.is_active(current_time) and not obj.detected:
                        for u in range(nUAVs):
                            if len(FLT_track[u]['X']) > 0:
                                if obj.is_in_fov(FLT_track[u]['X'][-1], 
                                               FLT_track[u]['Y'][-1],
                                               FLT_track[u]['Z'][-1],
                                               fov_radius):
                                    if not obj.detected:
                                        obj.detected = True
                                    if u not in obj.detected_by:
                                        obj.detected_by.append(u)
            
            iteration += 1
            
            # Mise à jour PX4 pour tous les UAVs
            await controller.update_all_from_simulation(FLT_track, FLT_conditions)
            
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
                print(f"  Décalage: {sync_delay:+.2f}s | Steps: B={params['bearing_step']}, V={params['speed_step']}")
                print(f"{'='*70}")
                
                for u in range(nUAVs):
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
                        
                        print(f"  {status}")
        
        # ========== FIN DE SIMULATION ==========
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
                metrics.powered_time[u] = phase_times[u]['powered']
        except Exception as e:
            print(f"⚠️  Erreur calcul métriques temps: {e}")
        
        # Calculer la distance minimale de séparation
        metrics.min_separation_distance = analyzer.check_min_separation(FLT_track, nUAVs)
        
        # Calculer les métriques de batterie
        for u in range(nUAVs):
            if len(FLT_track[u]['battery_capacity']) > 0:
                metrics.battery_remaining[u] = FLT_track[u]['battery_capacity'][-1]
                metrics.battery_consumed[u] = UAV_data['maximum_battery_capacity'] - metrics.battery_remaining[u]
        
        # Analyser les thermiques pour le scénario d'endurance
        if scenario == TestScenario.ENDURANCE and len(active_thermals) > 0:
            # Convertir le dictionnaire de thermiques en liste pour l'analyse
            thermals_list = list(active_thermals.values()) if isinstance(active_thermals, dict) else active_thermals
            thermal_analysis = analyzer.analyze_thermal_exploitation(thermals_list, FLT_track, nUAVs)
            metrics.thermals_detected = thermal_analysis['detected']
            metrics.thermals_exploited = thermal_analysis['exploited']
            metrics.thermals_rejected = thermal_analysis['rejected']
            metrics.thermals_per_uav = thermal_analysis['per_uav']
        
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
        filename = f"metrics_{scenario.value}_{nUAVs}uavs_{timestamp}.json"
        analyzer.save_metrics_to_file(metrics, filename)
        
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
        
        print("\n✓ Mission multi-UAV terminée avec succès!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption - Atterrissage d'urgence de tous les UAVs")
        await controller.land_all()
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        try:
            await controller.land_all()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(run_multi_uav_simulation())