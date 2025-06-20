# ---------- EXAMPLE ----------

import numpy as np
from GoToWP import gotoWaypoint
from compute import get_destination_from_range_and_bearing


nUAVs = 2

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

hub = dict()
hub['latitude'] = np.deg2rad(52.109601).tolist()
hub['longitude'] = np.deg2rad(-106.389644).tolist()
R = 3000.0
lat, lon = get_destination_from_range_and_bearing(hub, R, 0)
ubLAT = np.copy(lat).tolist()
lat, lon = get_destination_from_range_and_bearing(hub, R, pi)
lbLAT = np.copy(lat).tolist()
lat, lon = get_destination_from_range_and_bearing(hub, R, pi/2)
ubLON = np.copy(lon).tolist()
lat, lon = get_destination_from_range_and_bearing(hub, R, 3*pi/2)
lbLON = np.copy(lon).tolist()

params = dict()
params['working_floor'] = 600.0
params['latitude_lower_bound'] = lbLAT
params['latitude_upper_bound'] = ubLAT
params['longitude_lower_bound'] = lbLON
params['longitude_upper_bound'] = ubLON
params['altitude_lower_bound'] = 200.0
params['altitude_upper_bound'] = 1000.0
params['current_simulation_time'] = 0.0
params['time_step'] = 1
params['bearing_step'] = 10
params['speed_step'] = 10
params['safe_distance'] = 30.0
params['horizon_length'] = 100.0


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
FLT_track_keys = ['latitude', 'longitude', 'altitude', 'bearing', 'battery_capacity', 'flight_time', 'flight_mode']
FLT_conditions = {k: {} for k in range(nUAVs)}

for u in range(nUAVs):
    FLT_track[u] = dict()
    for keys in FLT_track_keys:
        FLT_track[u][keys] = []

    FLT_conditions[u] = dict()
    FLT_conditions[u]['airspeed'] = 13.0
    FLT_conditions[u]['weight'] = 0.0
    FLT_conditions[u]['flight_path_angle'] = 0.0
    FLT_conditions[u]['grav_accel'] = grav_accel
    FLT_conditions[u]['bank_angle'] = 0.0
    FLT_conditions[u]['airspeed_dot'] = 0.0
    FLT_conditions[u]['air_density'] = air_density
    FLT_conditions[u]['battery_capacity'] = UAV_data['maximum_battery_capacity']

for u in range(nUAVs):
    FLT_track[u]['latitude'].append(np.random.uniform(params['latitude_lower_bound'], params['latitude_upper_bound'], 1)[0].tolist())
    FLT_track[u]['longitude'].append(np.random.uniform(params['longitude_lower_bound'], params['longitude_upper_bound'], 1)[0].tolist())
    FLT_track[u]['altitude'].append(400.0)
    FLT_track[u]['bearing'].append(0.0)
    FLT_track[u]['battery_capacity'].append(UAV_data['maximum_battery_capacity'])
    FLT_track[u]['flight_time'].append(0.0)
    FLT_track[u]['flight_mode'].append('glide')

GOAL_WPs = dict()
GOAL_WPs['latitude'] = np.random.uniform(params['latitude_lower_bound'], params['latitude_upper_bound'], 10).tolist()
GOAL_WPs['longitude'] = np.random.uniform(params['longitude_lower_bound'], params['longitude_upper_bound'], 10).tolist()

# use trajectory from generator

Uidx = 0 # ID of UAV

FLT_track, FLT_conditions, current_wp_idx = gotoWaypoint(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, Uidx, params, UAV_data)