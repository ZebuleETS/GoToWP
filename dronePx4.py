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
from compute import compute_distance_cartesian, geographic_to_cartesian, cartesian_to_geographic
from trajectory import TrajectoryEvaluator, generate_all_trajectories, generate_random_obstacles, fix_trajectory, StraightLineTrajectory
from thermal import ThermalGenerator, ThermalMap, ThermalEvaluator


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
    
    async def connect(self):
        """Connexion à l'instance PX4 SITL"""
        connection_string = f"udpin://0.0.0.0:{self.connection_port}"
        print(f"[UAV {self.uav_id}] Connexion sur {connection_string}...")
        
        self.drone = System(mavsdk_server_address="127.0.0.1", mavsdk_server_port=self.mavsdk_port)
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
        
        #target_alt = 400
        #async for position in self.drone.telemetry.position():
        #    current_alt = position.relative_altitude_m
        #    print(f"[UAV {self.uav_id}] Altitude actuelle: {current_alt:.1f}m / {target_alt:.1f}m", end='\r')
        #    if current_alt >= target_alt * 0.9:
        #        print(f"[UAV {self.uav_id}] ✓ Altitude atteinte: {current_alt:.1f}m")
        #        break
    
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
        yaw_deg = np.degrees(bearing_rad)
        
        # Vitesse aérodynamique
        airspeed = FLT_conditions['airspeed']
        flight_path_angle = FLT_conditions['flight_path_angle']
        
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
        
        # Alternative: Convertir via ECEF si besoin de plus de précision
        # Pour les vitesses, on peut aussi faire la transformation directe
        # car c'est juste une rotation de coordonnées
        
        # Envoyer la commande combinée position + vitesse NED
        await self.drone.offboard.set_position_velocity_ned(
                PositionNedYaw(north_ned, east_ned, down_ned, yaw_deg),
                VelocityNedYaw(velocity_north_ned, velocity_east_ned, velocity_down_ned, yaw_deg)
            )
        #try:
        #    await self.drone.offboard.set_position_velocity_ned(
        #        PositionNedYaw(north_ned, east_ned, down_ned, yaw_deg),
        #        VelocityNedYaw(velocity_north_ned, velocity_east_ned, velocity_down_ned, yaw_deg)
        #    )
        #except Exception as e:
        #    print(f"[UAV {self.uav_id}] Erreur set_position_velocity_ned: {e}")
    
    async def start_offboard_mode(self):
        """Démarrer le mode offboard avec contrôle en vitesse"""
        print(f"[UAV {self.uav_id}] Démarrage mode offboard ...")
        await self.drone.offboard.set_position_ned(
               PositionNedYaw(0.0, 0.0, 0.0, 0.0)
            )
        
        try:
            await self.drone.offboard.start()
            print(f"[UAV {self.uav_id}] ✓ Mode offboard activé ")
            return True
        except OffboardError as error:
            print(f"[UAV {self.uav_id}] ✗ Erreur offboard: {error}")
            return False
    
    async def stop_offboard_mode(self):
        """Arrêter le mode offboard"""
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


async def run_multi_uav_simulation():
    """
    Fonction principale pour simulation multi-UAV avec PX4
    """
    print("="*70)
    print("SIMULATION MULTI-UAV PLANEUR AVEC THERMIQUES - PX4 SITL")
    print("="*70)
    
    # ========== PARAMÈTRES ==========
    nUAVs = int(input("\nNombre d'UAVs à simuler (1-10): ") or "3")
    nUAVs = max(1, min(10, nUAVs))  # Limiter entre 1 et 10
    
    print(f"\n✓ Configuration pour {nUAVs} UAVs")
    
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
    params['time_step'] = 4
    params['bearing_step'] = 10
    params['speed_step'] = 10
    params['safe_distance'] = 30.0
    params['horizon_length'] = 100.0
    
    # Génération d'obstacles
    obstacles = generate_random_obstacles(5, params)
    params['obstacles'] = obstacles
    print(f"✓ {len(obstacles)} obstacles générés")
    
    # ========== THERMIQUES ==========
    thermal_map = ThermalMap()
    thermal_generator = ThermalGenerator(params)
    thermal_evaluator = ThermalEvaluator(params, UAV_data)
    
    # Générer plus de thermiques pour multi-UAV
    num_thermals = max(3, nUAVs)
    active_thermals = thermal_generator.generate_random_thermals(
        num_thermals, obstacles, params['current_simulation_time']
    )
    print(f'✓ {len(active_thermals)} thermiques actives')
    
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
    
    print(f"\nInitialisation des {nUAVs} UAVs...")
    
    # ========== CONNEXION MULTI-UAV PX4 ==========
    controller = MultiUAVController(nUAVs, params, UAV_data)
    
    try:
        # Initialiser tous les UAVs
        await controller.initialize_all_uavs()
        
        print("\nConfiguration de la trajectoire initiale pour chaque UAV...")
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


            END_WPs[u]['X'].append(np.random.uniform(params['X_lower_bound'], params['X_upper_bound'], 1)[0].tolist())
            END_WPs[u]['Y'].append(np.random.uniform(params['Y_lower_bound'], params['Y_upper_bound'], 1)[0].tolist())
            END_WPs[u]['Z'].append(400.0)

            # Position initiale en utilisant pymap3d
            initial_lat = controller.bridges[u].home_position.latitude_deg
            initial_lon = controller.bridges[u].home_position.longitude_deg
            initial_alt = controller.bridges[u].simulation_origin['alt']
            
            # Convertir en coordonnées ENU locales par rapport à l'origine
            lat0 = controller.bridges[u].simulation_origin['lat']
            lon0 = controller.bridges[u].simulation_origin['lon']
            h0 = controller.bridges[u].simulation_origin['alt']
            
            initial_x, initial_y, initial_z = pm.geodetic2enu(
                initial_lat, initial_lon, initial_alt,
                lat0, lon0, h0
            )
            
            print(f"UAV {u} position initiale (E,N,U): ({initial_x:.1f}, {initial_y:.1f}, {initial_z:.1f})")
            
            FLT_track[u]['X'].append(initial_x)
            FLT_track[u]['Y'].append(initial_y)
            FLT_track[u]['Z'].append(400.0)
            FLT_track[u]['bearing'].append(0.0)
            FLT_track[u]['battery_capacity'].append(UAV_data['maximum_battery_capacity'])
            FLT_track[u]['flight_time'].append(0.0)
            FLT_track[u]['flight_mode'].append('glide')
            FLT_track[u]['in_evaluation'] = False
            FLT_track[u]['current_thermal_id'] = None
            FLT_track[u]['soaring_start_time'] = None

            # Génération de trajectoire
            startPoint = dict()
            startPoint['X'] = FLT_track[u]['X'][-1]
            startPoint['Y'] = FLT_track[u]['Y'][-1]
            startPoint['Z'] = FLT_track[u]['Z'][-1]
            startPoint['bearing'] = FLT_track[u]['bearing'][-1]

            straight_traj = StraightLineTrajectory(params, UAV_data)
            straight = straight_traj.generate_path(startPoint, END_WPs[u])
            optimal_trajectoires = fix_trajectory(straight, obstacles)
            
            GOAL_WPs[u]['X'] = optimal_trajectoires['X']
            GOAL_WPs[u]['Y'] = optimal_trajectoires['Y']
            GOAL_WPs[u]['Z'] = optimal_trajectoires['Z']
    
        current_wp_indices = {u: 1 for u in range(nUAVs)}
        current_eval_wp_indices = {u: 1 for u in range(nUAVs)}
        current_soar_wp_indices = {u: 1 for u in range(nUAVs)}
        
        # Décoller tous les UAVs
        await controller.arm_and_takeoff_all()
        
        # Démarrer le mode offboard pour tous
        await controller.start_offboard_all()
        
        print("\n" + "="*70)
        print("DÉBUT DE LA SIMULATION MULTI-UAV")
        print("="*70)
        
        # ========== BOUCLE PRINCIPALE ==========
        iteration = 0
        max_iterations = 2000
        
        total_decision_time = 0
        decision_calls = 0
        
        while iteration < max_iterations:
            # Vérifier si tous les UAVs ont terminé
            if all(current_wp_indices[u] >= len(GOAL_WPs[u]['X']) for u in range(nUAVs)):
                print("\n✓ Tous les UAVs ont atteint leurs objectifs!")
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
            
            # Mise à jour PX4 pour tous les UAVs
            await controller.update_all_from_simulation(FLT_track, FLT_conditions)
            
            # Affichage périodique
            if iteration % 20 == 0:
                avg_time = (total_decision_time / decision_calls) * 1000
                
                print(f"\n{'='*70}")
                print(f"[t={current_time:.1f}s | iter={iteration}] Temps décision: {avg_time:.2f}ms")
                print(f"{'='*70}")
                
                for u in range(nUAVs):
                    if len(FLT_track[u]['X']) > 0:
                        pos_str = f"({FLT_track[u]['X'][-1]:.0f}, {FLT_track[u]['Y'][-1]:.0f}, {FLT_track[u]['Z'][-1]:.0f})"
                        mode = FLT_track[u]['flight_mode'][-1]
                        battery = FLT_track[u]['battery_capacity'][-1]
                        airspeed = FLT_conditions[u]['airspeed']
                        
                        status = f"UAV {u}: {pos_str} | {mode} | V:{airspeed:.1f}m/s | Bat: {battery:.1f}"
                        
                        # Ajouter info thermique si applicable
                        if FLT_track[u]['current_thermal_id'] is not None:
                            status += f" | Therm: {FLT_track[u]['current_thermal_id']}"
                        
                        print(f"  {status}")
        
        # ========== FIN DE SIMULATION ==========
        print("\n" + "="*70)
        print("FIN DE LA SIMULATION MULTI-UAV")
        print("="*70)
        print(f"Temps total: {params['current_simulation_time']:.1f}s")
        print(f"Itérations: {iteration}")
        
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