# ---------- EXAMPLE ----------

import numpy as np
from GoToWP import gotoWaypointMulti
from compute import compute_distance_cartesian
from trajectory import TrajectoryEvaluator, generate_all_trajectories, generate_random_obstacles, StraightLineTrajectory, fix_trajectory
from thermal import ThermalGenerator, ThermalMap, ThermalEvaluator


nUAVs = 3

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
# Coefficient alpha (>1) pour augmenter la priorité d'un critère de décision
# En mode glide : augmente la priorité de minimiser la descente (C_sink)
# En mode engine : augmente la priorité de minimiser la consommation (C_energy)
params['alpha'] = 1.5  # Valeur recommandée: entre 1.0 et 3.0

# Définition des obstacles (polygones)
obstacles = generate_random_obstacles(5, params)
params['obstacles'] = obstacles

thermal_map = ThermalMap()
thermal_generator = ThermalGenerator(params)
thermal_evaluator = ThermalEvaluator(params, UAV_data)

active_thermals = thermal_generator.generate_random_thermals(3, obstacles, params['current_simulation_time'])
print(f'Active thermals: {active_thermals}')

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
    trajectoires = generate_all_trajectories(startPoint,END_WPs[u], params, UAV_data, obstacles)
    optimal_trajectoires = evaluator.evaluate_trajectories(trajectoires)
    GOAL_WPs[u]['X'] = optimal_trajectoires['X']
    GOAL_WPs[u]['Y'] = optimal_trajectoires['Y']
    GOAL_WPs[u]['Z'] = optimal_trajectoires['Z']


current_wp_indices = dict()
current_eval_wp_indices = dict()
current_soar_wp_indices = dict()
for u in range(nUAVs):
    current_wp_indices[u] = 1  # Initialize the current waypoint index for each UAV
    current_eval_wp_indices[u] = 1  # Initialize the current evaluation waypoint index for each UAV
    current_soar_wp_indices[u] = 1  # Initialize the current soaring waypoint index for each UAV

print(f'Fin: {GOAL_WPs[0]["X"][-1]}, {GOAL_WPs[0]["Y"][-1]}, {GOAL_WPs[0]["Z"][-1]}')
print(f'Début: {startPoint}')
D2 = compute_distance_cartesian(startPoint, GOAL_WPs[0])[-1]
print(f'distance: {D2}')

# Ajoutez ceci au début de votre boucle while pour mesurer les performances
import time

total_decision_time = 0
decision_calls = 0

while True:
    # Check if all UAVs have reached their goal waypoints
    if all(current_wp_indices[u] >= len(GOAL_WPs[u]['X']) for u in range(nUAVs)):
        break

    params['current_simulation_time'] += params['time_step']
    current_time = params['current_simulation_time']    

    # Call the gotoWaypointMulti function to update the flight track and conditions
    # Mesurer le temps de la fonction gotoWaypointMulti
    start_time = time.perf_counter()
    FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices, SOAR_WPs, current_soar_wp_indices = gotoWaypointMulti(
        FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data,
        current_wp_indices, current_eval_wp_indices, current_soar_wp_indices, thermal_map, thermal_evaluator, EVAL_WPs, active_thermals, SOAR_WPs
    )
    end_time = time.perf_counter()
    
    total_decision_time += (end_time - start_time)
    decision_calls += 1
    
    if decision_calls % 10 == 0:  # Afficher toutes les 10 itérations
        avg_time = (total_decision_time / decision_calls) * 1000
        print(f"Temps moyen par décision: {avg_time:.2f} ms")
        
        
    print(f"Temps: {current_time}s - UAV 0 - Alt: {FLT_track[0]['Z'][-1]:.1f}m, Mode: {FLT_track[0]['flight_mode'][-1]}")
    print(f"Temps: {current_time}s - UAV 1 - Alt: {FLT_track[1]['Z'][-1]:.1f}m, Mode: {FLT_track[1]['flight_mode'][-1]}")
    print(f"Temps: {current_time}s - UAV 2 - Alt: {FLT_track[2]['Z'][-1]:.1f}m, Mode: {FLT_track[2]['flight_mode'][-1]}")
    
    # Afficher les gains d'altitude si en évaluation ou soaring
    if FLT_track[0]['in_evaluation'] and 'evaluation_start_altitude' in FLT_track[0]:
        altitude_gain = FLT_track[0]['Z'][-1] - FLT_track[0]['evaluation_start_altitude']
        print(f"  Gain d'altitude depuis début évaluation: {altitude_gain:.1f}m")
    elif FLT_track[0]['flight_mode'][-1] == 'soaring' and FLT_track[0].get('soaring_start_time'):
        # Calculer le gain depuis le début du soaring
        soaring_duration = current_time - FLT_track[0]['soaring_start_time']
        print(f"  Durée soaring: {soaring_duration:.0f}s")