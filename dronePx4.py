"""
Integration de votre simulation de planeur avec PX4 SITL
Version mise à jour avec obstacles polygonaux et mode soaring
"""

import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw
from mavsdk.telemetry import LandedState
import time

# Importer vos modules existants
from GoToWP import gotoWaypointMulti
from compute import compute_distance_cartesian
from trajectory import TrajectoryEvaluator, generate_all_trajectories, generate_random_obstacles
from thermal import ThermalGenerator, ThermalMap, ThermalEvaluator


class PX4SITLBridge:
    """
    Pont entre votre simulation et PX4 SITL
    Convertit vos coordonnées XYZ en commandes PX4
    """
    
    def __init__(self, params, UAV_data):
        self.params = params
        self.UAV_data = UAV_data
        self.drone = System()
        self.is_connected = False
        self.home_position = None
        self.simulation_origin = None
        
        # Taux de mise à jour (Hz)
        self.update_rate = 10  # 10Hz pour fixed-wing
        
    async def connect(self, connection_string="udp://:14540"):
        """Connexion à PX4 SITL"""
        print(f"Connexion à PX4 SITL sur {connection_string}...")
        await self.drone.connect(system_address=connection_string)
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("✓ Drone connecté!")
                self.is_connected = True
                break
        
        # Attendre le fix GPS
        print("Attente du fix GPS...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("✓ GPS fix obtenu!")
                break
        
        # Stocker la position home
        async for position in self.drone.telemetry.position():
            self.home_position = position
            print(f"✓ Position home: {position.latitude_deg:.6f}, {position.longitude_deg:.6f}")
            break
        
        # Définir l'origine de la simulation
        self.simulation_origin = {
            'lat': self.home_position.latitude_deg,
            'lon': self.home_position.longitude_deg,
            'alt': self.home_position.absolute_altitude_m
        }
    
    def xyz_to_ned(self, x, y, z):
        """
        Convertir XYZ (votre système) vers NED (PX4)
        X = Est, Y = Nord, Z = Altitude AGL
        NED: North, East, Down
        """
        north = y
        east = x
        down = -(z - self.params['working_floor'])  # Relatif au working floor
        return north, east, down
    
    def ned_to_gps(self, north, east, down):
        """
        Convertir NED vers coordonnées GPS
        Pour petites distances (~10km), approximation linéaire
        """
        lat_offset = north / 111320.0  # mètres vers degrés
        lon_offset = east / (111320.0 * np.cos(np.radians(self.simulation_origin['lat'])))
        
        lat = self.simulation_origin['lat'] + lat_offset
        lon = self.simulation_origin['lon'] + lon_offset
        alt = self.simulation_origin['alt'] - down  # Down est négatif vers le haut
        
        return lat, lon, alt
    
    async def arm_and_takeoff(self):
        """Armement et décollage"""
        print("\n--- Armement et Décollage ---")
        
        # Armer
        print("Armement...")
        await self.drone.action.arm()
        await asyncio.sleep(2)
        
        # Décollage fixed-wing
        print("Décollage...")
        await self.drone.action.takeoff()
        
        # Attendre d'atteindre l'altitude de croisière
        target_alt = self.params['working_floor']
        print(f"Montée vers {target_alt}m...")
        
        async for position in self.drone.telemetry.position():
            current_alt = position.relative_altitude_m
            print(f"  Altitude: {current_alt:.1f}m / {target_alt}m", end='\r')
            if current_alt >= target_alt * 0.9:
                print(f"\n✓ Altitude de croisière atteinte: {current_alt:.1f}m")
                break
            await asyncio.sleep(0.5)
    
    async def send_position_ned(self, north, east, down, yaw=0.0):
        """
        Envoyer une commande de position en NED
        Utilisé en mode offboard
        """
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(north, east, down, yaw)
        )
    
    async def send_velocity_ned(self, velocity_north, velocity_east, velocity_down, yaw_rate=0.0):
        """
        Envoyer une commande de vélocité en NED
        Utile pour contrôle plus dynamique en mode soaring
        """
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(velocity_north, velocity_east, velocity_down, yaw_rate)
        )
    
    async def update_from_simulation_state(self, FLT_track, u):
        """
        Mettre à jour PX4 avec l'état actuel de votre simulation
        """
        if len(FLT_track[u]['X']) == 0:
            return
        
        # Position actuelle de la simulation
        x = FLT_track[u]['X'][-1]
        y = FLT_track[u]['Y'][-1]
        z = FLT_track[u]['Z'][-1]
        bearing = FLT_track[u]['bearing'][-1]
        
        # Convertir et envoyer à PX4
        north, east, down = self.xyz_to_ned(x, y, z)
        
        # Convertir bearing en yaw (radians)
        yaw = np.radians(bearing)
        
        await self.send_position_ned(north, east, down, yaw)
    
    async def start_offboard_mode(self):
        """Démarrer le mode offboard pour contrôle direct"""
        print("Démarrage du mode offboard...")
        
        # Envoyer une position initiale
        await self.send_position_ned(0.0, 0.0, -self.params['working_floor'])
        
        try:
            await self.drone.offboard.start()
            print("✓ Mode offboard activé")
        except OffboardError as error:
            print(f"✗ Erreur mode offboard: {error}")
            return False
        
        return True
    
    async def stop_offboard_mode(self):
        """Arrêter le mode offboard"""
        try:
            await self.drone.offboard.stop()
            print("✓ Mode offboard désactivé")
        except OffboardError as error:
            print(f"Erreur arrêt offboard: {error}")
    
    async def return_and_land(self):
        """Retour et atterrissage"""
        print("\n--- Retour à la Base ---")
        await self.stop_offboard_mode()
        await self.drone.action.return_to_launch()
        
        # Attendre l'atterrissage
        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                print("✓ Atterrissage réussi")
                break
            await asyncio.sleep(1)
    
    async def get_real_position(self):
        """
        Récupérer la position réelle du drone depuis PX4
        Utile pour feedback dans votre simulation
        """
        async for position in self.drone.telemetry.position():
            return {
                'lat': position.latitude_deg,
                'lon': position.longitude_deg,
                'alt': position.absolute_altitude_m,
                'alt_rel': position.relative_altitude_m
            }
    
    async def monitor_telemetry(self, duration=5):
        """Monitorer la télémétrie"""
        print(f"\n--- Télémétrie ({duration}s) ---")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            pos_task = self.drone.telemetry.position().__anext__()
            vel_task = self.drone.telemetry.velocity_ned().__anext__()
            att_task = self.drone.telemetry.attitude_euler().__anext__()
            
            position = await pos_task
            velocity = await vel_task
            attitude = await att_task
            
            speed = np.sqrt(velocity.north_m_s**2 + velocity.east_m_s**2)
            print(f"\rAlt: {position.relative_altitude_m:.1f}m | "
                  f"Speed: {speed:.1f}m/s | "
                  f"Heading: {attitude.yaw_deg:.0f}° | "
                  f"Pitch: {attitude.pitch_deg:.1f}°", end='')
            
            await asyncio.sleep(0.2)
        print()


async def run_simulation_with_px4():
    """
    Fonction principale qui combine votre simulation avec PX4
    """
    print("="*70)
    print("Simulation de Planeur avec Thermiques - Intégration PX4 SITL")
    print("="*70)
    
    # ========== VOS PARAMÈTRES EXISTANTS ==========
    nUAVs = 1
    
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
    params['time_step'] = 1
    params['bearing_step'] = 10
    params['speed_step'] = 10
    params['safe_distance'] = 30.0
    params['horizon_length'] = 100.0
    
    # Génération d'obstacles polygonaux
    obstacles = generate_random_obstacles(5, params)
    params['obstacles'] = obstacles
    print(f"✓ {len(obstacles)} obstacles polygonaux générés")
    
    # ========== INITIALISATION THERMIQUES ==========
    thermal_map = ThermalMap()
    thermal_generator = ThermalGenerator(params)
    thermal_evaluator = ThermalEvaluator(params, UAV_data)
    
    active_thermals = thermal_generator.generate_random_thermals(3, obstacles, params['current_simulation_time'])
    print(f'✓ {len(active_thermals)} thermiques actives générées')
    
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
    
    # ========== INITIALISATION UAV ==========
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
    
    for u in range(nUAVs):
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
        
        # Waypoint final (première thermique)
        END_WPs[u]['X'].append(active_thermals[0].x)
        END_WPs[u]['Y'].append(active_thermals[0].y)
        END_WPs[u]['Z'].append(400.0)
        
        # Position initiale aléatoire
        FLT_track[u]['X'].append(np.random.uniform(params['X_lower_bound'], params['X_upper_bound'], 1)[0])
        FLT_track[u]['Y'].append(np.random.uniform(params['Y_lower_bound'], params['Y_upper_bound'], 1)[0])
        FLT_track[u]['Z'].append(400.0)
        FLT_track[u]['bearing'].append(0.0)
        FLT_track[u]['battery_capacity'].append(UAV_data['maximum_battery_capacity'])
        FLT_track[u]['flight_time'].append(0.0)
        FLT_track[u]['flight_mode'].append('glide')
        FLT_track[u]['in_evaluation'] = False
        FLT_track[u]['current_thermal_id'] = None
        FLT_track[u]['soaring_start_time'] = None
        
        # Génération de trajectoire initiale
        evaluator = TrajectoryEvaluator(params, UAV_data, FLT_conditions[u])
        startPoint = dict()
        startPoint['X'] = FLT_track[u]['X'][-1]
        startPoint['Y'] = FLT_track[u]['Y'][-1]
        startPoint['Z'] = FLT_track[u]['Z'][-1]
        startPoint['bearing'] = FLT_track[u]['bearing'][-1]
        
        trajectoires = generate_all_trajectories(startPoint, END_WPs[u], params, UAV_data, obstacles)
        optimal_trajectoires = evaluator.evaluate_trajectories(trajectoires)
        GOAL_WPs[u]['X'] = optimal_trajectoires['X']
        GOAL_WPs[u]['Y'] = optimal_trajectoires['Y']
        GOAL_WPs[u]['Z'] = optimal_trajectoires['Z']
    
    current_wp_indices = {u: 1 for u in range(nUAVs)}
    current_eval_wp_indices = {u: 1 for u in range(nUAVs)}
    current_soar_wp_indices = {u: 1 for u in range(nUAVs)}
    
    print(f'✓ Position de départ: ({startPoint["X"]:.1f}, {startPoint["Y"]:.1f}, {startPoint["Z"]:.1f})')
    print(f'✓ Objectif initial: ({GOAL_WPs[0]["X"][-1]:.1f}, {GOAL_WPs[0]["Y"][-1]:.1f}, {GOAL_WPs[0]["Z"][-1]:.1f})')
    
    D2 = compute_distance_cartesian(startPoint, GOAL_WPs[0])[-1]
    print(f'✓ Distance à parcourir: {D2:.1f}m')
    
    # ========== CONNEXION PX4 ==========
    px4_bridge = PX4SITLBridge(params, UAV_data)
    
    try:
        await px4_bridge.connect()
        await px4_bridge.arm_and_takeoff()
        
        # Démarrer le mode offboard
        if not await px4_bridge.start_offboard_mode():
            print("✗ Impossible de démarrer le mode offboard")
            return
        
        print("\n" + "="*70)
        print("DÉBUT DE LA SIMULATION")
        print("="*70)
        
        # ========== BOUCLE DE SIMULATION PRINCIPALE ==========
        iteration = 0
        max_iterations = 2000  # Limite de sécurité
        
        # Métriques de performance
        total_decision_time = 0
        decision_calls = 0
        
        while iteration < max_iterations:
            # Vérifier si tous les UAVs ont atteint leurs objectifs
            if all(current_wp_indices[u] >= len(GOAL_WPs[u]['X']) for u in range(nUAVs)):
                print("\n✓ Tous les waypoints ont été atteints!")
                break
            
            params['current_simulation_time'] += params['time_step']
            current_time = params['current_simulation_time']
            iteration += 1
            
            # Mesurer le temps de décision
            start_time = time.perf_counter()
            
            # Mise à jour de la simulation
            FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices, SOAR_WPs, current_soar_wp_indices = gotoWaypointMulti(
                FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data,
                current_wp_indices, current_eval_wp_indices, current_soar_wp_indices, 
                thermal_map, thermal_evaluator, EVAL_WPs, active_thermals, SOAR_WPs
            )
            
            end_time = time.perf_counter()
            total_decision_time += (end_time - start_time)
            decision_calls += 1
            
            # Mise à jour de PX4 avec la nouvelle position
            await px4_bridge.update_from_simulation_state(FLT_track, 0)
            
            # Affichage périodique
            if iteration % 10 == 0:
                avg_time = (total_decision_time / decision_calls) * 1000
                
                print(f"\n[t={current_time:.1f}s | iter={iteration}]")
                print(f"  Position: ({FLT_track[0]['X'][-1]:.1f}, {FLT_track[0]['Y'][-1]:.1f}, {FLT_track[0]['Z'][-1]:.1f})")
                print(f"  Altitude: {FLT_track[0]['Z'][-1]:.1f}m")
                print(f"  Mode: {FLT_track[0]['flight_mode'][-1]}")
                print(f"  Batterie: {FLT_track[0]['battery_capacity'][-1]:.2f}/{UAV_data['maximum_battery_capacity']:.2f}")
                print(f"  Bearing: {FLT_track[0]['bearing'][-1]:.1f}°")
                print(f"  Temps décision moyen: {avg_time:.2f}ms")
                
                # Afficher gains d'altitude si en évaluation
                if FLT_track[0]['in_evaluation'] and 'evaluation_start_altitude' in FLT_track[0]:
                    altitude_gain = FLT_track[0]['Z'][-1] - FLT_track[0]['evaluation_start_altitude']
                    print(f"  Gain altitude (évaluation): {altitude_gain:.1f}m")
                
                # Afficher durée soaring
                elif FLT_track[0]['flight_mode'][-1] == 'soaring' and FLT_track[0].get('soaring_start_time'):
                    soaring_duration = current_time - FLT_track[0]['soaring_start_time']
                    print(f"  Durée soaring: {soaring_duration:.0f}s")
            
            # Petite pause pour synchronisation
            await asyncio.sleep(1.0 / px4_bridge.update_rate)
        
        print("\n" + "="*70)
        print("FIN DE LA SIMULATION")
        print("="*70)
        print(f"Temps total: {params['current_simulation_time']:.1f}s")
        print(f"Itérations: {iteration}")
        print(f"Position finale: ({FLT_track[0]['X'][-1]:.1f}, {FLT_track[0]['Y'][-1]:.1f}, {FLT_track[0]['Z'][-1]:.1f})")
        print(f"Altitude finale: {FLT_track[0]['Z'][-1]:.1f}m")
        print(f"Batterie restante: {FLT_track[0]['battery_capacity'][-1]:.2f}/{UAV_data['maximum_battery_capacity']:.2f}")
        print(f"Temps de vol: {FLT_track[0]['flight_time'][-1]:.1f}s")
        
        # Afficher statistiques des modes de vol
        mode_counts = {}
        for mode in FLT_track[0]['flight_mode']:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print("\nStatistiques des modes de vol:")
        for mode, count in mode_counts.items():
            percentage = (count / len(FLT_track[0]['flight_mode'])) * 100
            print(f"  {mode}: {count} ({percentage:.1f}%)")
        
        # Monitorer télémétrie finale
        await px4_bridge.monitor_telemetry(duration=5)
        
        # Retour et atterrissage
        await px4_bridge.return_and_land()
        
        print("\n✓ Mission terminée avec succès!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur - Atterrissage d'urgence")
        await px4_bridge.return_and_land()
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        try:
            await px4_bridge.return_and_land()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(run_simulation_with_px4())
