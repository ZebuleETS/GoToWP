# ---------- EXAMPLE ----------

import numpy as np
from GoToWP import gotoWaypointMulti
from compute import compute_distance_cartesian
from trajectory import StraightLineTrajectory, TrajectoryEvaluator, generate_all_trajectories
from thermal import ThermalGenerator, ThermalMap, ThermalEvaluator, detect_thermal_at_position


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
        'X': 1000,        # Position X du centre (m)
        'Y': 1500,        # Position Y du centre (m)
        'radius': 200,    # Rayon du cylindre (m)
        'Z_min': 0,     # Altitude minimale de l'obstacle (m)
        'Z_max': 800      # Altitude maximale de l'obstacle (m)
    },
    {
        'X': 3000,
        'Y': 2000,
        'radius': 350,
        'Z_min': 0,
        'Z_max': 500
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

active_thermals = thermal_generator.generate_random_thermals(3, params['current_simulation_time'])

#test du soaring
#for thermal_id, thermal in active_thermals.items():
#    detected_thermal_id = thermal_id
#    thermal_map.add_thermal_detection(detected_thermal_id, thermal, params['current_simulation_time'])
#    thermal_map.change_thermal_status(detected_thermal_id, True, True)

print(f'Active thermals strength: {active_thermals}')
print(f'Active thermals: {[thermal.get_strength() for thermal in active_thermals.values()]}')
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
FLT_track_keys = ['X', 'Y', 'Z', 'bearing', 'battery_capacity', 'flight_time', 'flight_mode', 'in_evaluation', 'evaluation_start_altitude', 'current_thermal_id', 'soaring_start_time']
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

    #END_WPs[u]['X'].append(np.random.uniform(params['X_lower_bound'], params['X_upper_bound'], 1)[0].tolist())
    #END_WPs[u]['Y'].append(np.random.uniform(params['Y_lower_bound'], params['Y_upper_bound'], 1)[0].tolist())
    END_WPs[u]['X'].append(active_thermals[0].x + 500.0)
    END_WPs[u]['Y'].append(active_thermals[0].y + 500.0)
    END_WPs[u]['Z'].append(400.0)

    FLT_track[u]['X'].append(active_thermals[0].x - 400.0)
    FLT_track[u]['Y'].append(active_thermals[0].y - 400.0)
    FLT_track[u]['Z'].append(400.0)
    FLT_track[u]['bearing'].append(0.0)
    FLT_track[u]['battery_capacity'].append(UAV_data['maximum_battery_capacity'])
    FLT_track[u]['flight_time'].append(0.0)
    FLT_track[u]['flight_mode'].append('glide')
    FLT_track[u]['in_evaluation'] = False
    FLT_track[u]['current_thermal_id'] = None
    FLT_track[u]['soaring_start_time'] = 0.0


    startPoint = dict()
    startPoint['X'] = FLT_track[u]['X'][-1]
    startPoint['Y'] = FLT_track[u]['Y'][-1]
    startPoint['Z'] = FLT_track[u]['Z'][-1]
    startPoint['bearing'] = FLT_track[u]['bearing'][-1]
    params['num_points'] = 20
    straight_traj = StraightLineTrajectory(params, UAV_data)
    straight = straight_traj.generate_path(startPoint, END_WPs[u])
    GOAL_WPs[u]['X'] = straight['X']
    GOAL_WPs[u]['Y'] = straight['Y']
    GOAL_WPs[u]['Z'] = straight['Z']


current_wp_indices = dict()
current_eval_wp_indices = dict()
for u in range(nUAVs):
    current_wp_indices[u] = 1  # Initialize the current waypoint index for each UAV
    current_eval_wp_indices[u] = 1  # Initialize the current evaluation waypoint index for each UAV

print(f'Fin: {GOAL_WPs[0]["X"][-1]}, {GOAL_WPs[0]["Y"][-1]}, {GOAL_WPs[0]["Z"][-1]}')
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
        # si une thermique est détectée, on l'ajoute à la carte des thermiques
        if detected_thermal_id is not None:
            if detected_thermal_id not in thermal_map.detected_thermals:
                thermal_map.add_thermal_detection(detected_thermal_id, active_thermals[detected_thermal_id], current_time)
                
                FLT_track[u]['evaluation_start_altitude'] = current_pos['Z']
                
                # Générér les Wps d'évaluation pour la thermique détectée
                trajectoires = thermal_map.generate_evaluation_waypoints(current_pos, detected_thermal_id)
                FLT_track[u]['in_evaluation'] = True
                FLT_track[u]['current_thermal_id'] = detected_thermal_id
                EVAL_WPs[u]['X'] = trajectoires['X']
                EVAL_WPs[u]['Y'] = trajectoires['Y']
                EVAL_WPs[u]['Z'] = trajectoires['Z']
                # Add circle to obstacles
                thermal = active_thermals[detected_thermal_id]
                evaluation_obstacle = {
                    'X': thermal.x,
                    'Y': thermal.y,
                    'radius': thermal.radius,
                    'Z_min': current_pos['Z'] - 50,  # 50m en dessous
                    'Z_max': current_pos['Z'] + 200,  # 200m au-dessus
                    'type': 'evaluation_zone',
                    'uav_id': u
                }
                # Add the evaluation obstacle to the list of obstacles
                if 'obstacles' not in params:
                    params['obstacles'] = []
                params['obstacles'].append(evaluation_obstacle)
            else:
                thermal = thermal_map.detected_thermals[detected_thermal_id]['thermal']
                if thermal.is_active(current_time) and FLT_track[u]['current_thermal_id'] != detected_thermal_id and (thermal_map.detected_thermals[detected_thermal_id]['evaluated'] and thermal_map.detected_thermals[detected_thermal_id]['alt_gain']):
                    # Si la thermique est active et a été évaluée, on peut l'utiliser pour le soaring
                    FLT_track[u]['current_thermal_id'] = detected_thermal_id
                    FLT_track[u]['flight_mode'].append('soaring')
                    if len(FLT_track[u]['flight_mode']) == 1 or FLT_track[u]['flight_mode'][-2] != 'soaring':
                        FLT_track[u]['soaring_start_time'] = current_time


    # Nettoyer les obstacles temporaires des UAVs qui ne sont plus en évaluation
    params['obstacles'] = [obs for obs in params['obstacles'] if obs.get('uav_id') is None or FLT_track[obs['uav_id']]['in_evaluation']]
    # Call the gotoWaypointMulti function to update the flight track and conditions
    FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices = gotoWaypointMulti(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data, current_wp_indices, current_eval_wp_indices, thermal_map, thermal_evaluator, EVAL_WPs)
    
    print(f"Temps: {current_time}s - UAV 0 - Alt: {FLT_track[0]['Z'][-1]:.1f}m, Mode: {FLT_track[0]['flight_mode'][-1]}")
    
    # Afficher les gains d'altitude si en évaluation ou soaring
    if FLT_track[0]['in_evaluation'] and 'evaluation_start_altitude' in FLT_track[0]:
        altitude_gain = FLT_track[0]['Z'][-1] - FLT_track[0]['evaluation_start_altitude']
        print(f"  Gain d'altitude depuis début évaluation: {altitude_gain:.1f}m")
    elif FLT_track[0]['flight_mode'][-1] == 'soaring' and FLT_track[0].get('soaring_start_time'):
        # Calculer le gain depuis le début du soaring
        soaring_duration = current_time - FLT_track[0]['soaring_start_time']
        print(f"  Durée soaring: {soaring_duration:.0f}s")