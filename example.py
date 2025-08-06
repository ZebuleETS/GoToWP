# ---------- EXAMPLE ----------

from math import pi
import numpy as np
from GoToWP import gotoWaypointMulti
from compute import compute_distance, compute_distance_cartesian, get_destination_from_range_and_bearing
from trajectory import TrajectoryEvaluator, generate_all_trajectories
from thermal import ThermalGenerator, ThermalMap, ThermalEvaluator, ThermalExploiter, detect_thermal_at_position


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

# Définition des obstacles (cylindres)
obstacles = [
    {
        'x': 1000,        # Position X du centre (m)
        'y': 1500,        # Position Y du centre (m)
        'radius': 200,    # Rayon du cylindre (m)
        'z_min': 0,     # Altitude minimale de l'obstacle (m)
        'z_max': 800      # Altitude maximale de l'obstacle (m)
    },
    {
        'x': 3000,
        'y': 2000,
        'radius': 350,
        'z_min': 0,
        'z_max': 500
    }
]

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
params['obstacles'] = obstacles

thermal_map = ThermalMap()
thermal_generator = ThermalGenerator(params)
thermal_evaluator = ThermalEvaluator(params, UAV_data)
thermal_exploiter = ThermalExploiter(params, UAV_data)

active_thermals = thermal_generator.generate_random_thermals(3, params['current_simulation_time'])


ACC_SEA_LEVEL = 9.80665
T_SEA_LEVEL = 288.15
RHO_SEA_LEVEL = 1.225
MEAN_EARTH_RADIUS = 6371009
TROPO_LAPSE_RATE = -0.0065
R = 287.058
grav_accel = ACC_SEA_LEVEL * (MEAN_EARTH_RADIUS /(MEAN_EARTH_RADIUS + params['working_floor']))
T_fin = T_SEA_LEVEL + TROPO_LAPSE_RATE * (params['working_floor'])
air_density = RHO_SEA_LEVEL * (T_fin / T_SEA_LEVEL)**(-grav_accel / (TROPO_LAPSE_RATE * R) - 1)

FLT_track = {k: {} for k in range(nUAVs)}
FLT_track_keys = ['X', 'Y', 'Z', 'bearing', 'battery_capacity', 'flight_time', 'flight_mode', 'in_evaluation', 'current_thermal_id', 'soaring_start_time']
FLT_conditions = {k: {} for k in range(nUAVs)}
END_WPs = {k: {} for k in range(nUAVs)}
WPs_keys = ['X', 'Y', 'Z']
GOAL_WPs = {k: {} for k in range(nUAVs)}
EVAL_WPs = {k: {} for k in range(nUAVs)}

for u in range(nUAVs):
    FLT_track[u] = dict()
    for keys in FLT_track_keys:
        FLT_track[u][keys] = []
    for keys in WPs_keys:
        END_WPs[u][keys] = []
        GOAL_WPs[u][keys] = []
        EVAL_WPs[u][keys] = []
        
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

    FLT_track[u]['X'].append(np.random.uniform(params['X_lower_bound'], params['X_upper_bound'], 1)[0].tolist())
    FLT_track[u]['Y'].append(np.random.uniform(params['Y_lower_bound'], params['Y_upper_bound'], 1)[0].tolist())
    FLT_track[u]['Z'].append(400.0)
    FLT_track[u]['bearing'].append(0.0)
    FLT_track[u]['battery_capacity'].append(UAV_data['maximum_battery_capacity'])
    FLT_track[u]['flight_time'].append(0.0)
    FLT_track[u]['flight_mode'].append('glide')
    FLT_track[u]['in_evaluation'] = False
    FLT_track[u]['current_thermal_id'] = None
    FLT_track[u]['soaring_start_time'] = None

    evaluator = TrajectoryEvaluator(params, UAV_data, FLT_conditions[u])
    startPoint = dict()
    startPoint['X'] = FLT_track[u]['X'][-1]
    startPoint['Y'] = FLT_track[u]['Y'][-1]
    startPoint['Z'] = FLT_track[u]['Z'][-1]
    startPoint['bearing'] = FLT_track[u]['bearing'][-1]
    trajectoires = generate_all_trajectories(startPoint,END_WPs[u], params, UAV_data)
    optimal_trajectoires = evaluator.evaluate_trajectories(trajectoires)
    GOAL_WPs[u]['X'] = optimal_trajectoires['X']
    GOAL_WPs[u]['Y'] = optimal_trajectoires['Y']
    GOAL_WPs[u]['Z'] = optimal_trajectoires['Z']


current_wp_indices = dict()
current_eval_wp_indices = dict()
for u in range(nUAVs):
    current_wp_indices[u] = 1  # Initialize the current waypoint index for each UAV
    current_eval_wp_indices[u] = 0  # Initialize the current evaluation waypoint index for each UAV

print(f'Fin: {GOAL_WPs[0]}')
print(f'Début: {startPoint}')
D2 = compute_distance_cartesian(startPoint, GOAL_WPs[0])[-1]
print(f'distance: {D2}')

while True:
    # Check if all UAVs have reached their goal waypoints
    if all(current_wp_indices[u] >= len(GOAL_WPs[u]['X']) for u in range(nUAVs)):
        break

    params['current_simulation_time'] += params['time_step']
    current_time = params['current_simulation_time']    

    for u in range(nUAVs):
        current_pos = {
            'X': FLT_track[u]['X'][-1],
            'Y': FLT_track[u]['Y'][-1],
            'Z': FLT_track[u]['Z'][-1]
        }

        # Détection d'un nouveau thermique
        detected_thermal_id = detect_thermal_at_position(current_pos, active_thermals, current_time)
        
        # si des thermiques sont détectés, on les ajoute à la carte des thermiques
        if detected_thermal_id is not None:
            if detected_thermal_id not in thermal_map.detected_thermals:
                thermal_map.add_thermal_detection(detected_thermal_id, active_thermals[detected_thermal_id], current_time)

            # Générér les Wps d'évaluation pour les thermiques détectés
            trajectoires = thermal_map.generate_evaluation_waypoints(current_pos, [detected_thermal_id])
            FLT_track[u]['in_evaluation'] = True
            FLT_track[u]['current_thermal_id'] = detected_thermal_id
            EVAL_WPs[u]['X'] = trajectoires['X']
            EVAL_WPs[u]['Y'] = trajectoires['Y']
            EVAL_WPs[u]['Z'] = trajectoires['Z']
            # Add circle to obstacles
            thermal = active_thermals[detected_thermal_id]
            evaluation_obstacle = {
                'x': thermal.x,
                'y': thermal.y,
                'radius': thermal.radius,
                'z_min': current_pos['Z'] - 50,  # 50m en dessous
                'z_max': current_pos['Z'] + 200,  # 200m au-dessus
                'type': 'evaluation_zone',
                'uav_id': u
            }
            # Add the evaluation obstacle to the list of obstacles
            if 'obstacles' not in params:
                params['obstacles'] = []
            params['obstacles'].append(evaluation_obstacle)
    
    # Nettoyer les obstacles temporaires des UAVs qui ne sont plus en évaluation
    params['obstacles'] = [obs for obs in params['obstacles'] if obs.get('uav_id') is None or FLT_track[obs['uav_id']]['in_evaluation']]
    # Call the gotoWaypointMulti function to update the flight track and conditions
    FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices = gotoWaypointMulti(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data, current_wp_indices, current_eval_wp_indices, thermal_map, thermal_evaluator, EVAL_WPs)
    #print(FLT_conditions[0]['airspeed'])
    #print(FLT_track[0]['X'])
    #print(FLT_track[0]['Y'])
    #print(FLT_track[0]['Z'])
    #print(current_wp_indices)
    #print(current_eval_wp_indices)