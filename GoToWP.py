from math import cos, sin, pi, sqrt, atan, atan2
from typing import Dict, Tuple
import numpy as np
import warnings
from compute import (
    cartesian_to_geographic,
    geographic_to_cartesian,
    get_current_flight_data,
    get_power_consumption,
    get_sink_rate,
    get_destination_from_range_and_bearing
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#not used
def update_remaining_energy(flight_conditions, power_consumption, time_step):
    """
    Updates the remaining battery energy after a given flight duration.

    Args:
        flight_conditions (dict): Dictionary of current flight conditions.
        power_consumption (float): Power consumption (W).
        time_step (float or list): Flight duration(s) (s).

    Returns:
        list: List of remaining energy (Wh) after each time step.
    """
    remaining_energy = []

    if isinstance(time_step, float):
        time_step = [time_step]

    n = len(time_step)

    for i in range(n):
        energy_cost = power_consumption * (time_step[i] / 3600)
        remaining_energy.append(flight_conditions['battery_capacity'] - energy_cost)

    return remaining_energy


def compute_distance(pos, dest):
    """
    Calculates the 3D (horizontal and vertical) distance between two geographic points.

    Args:
        pos (dict): Start position (latitude, longitude, altitude).
        dest (dict): Destination position (latitude, longitude, altitude).

    Returns:
        list: List of distances (meters) for each destination.
    """
    if isinstance(pos['altitude'], list):
        pos_alt = pos['altitude'][-1]
    else:
        pos_alt = pos['altitude']

    dest_alt = dest['altitude']
    vert_dist = []

    if isinstance(dest_alt, float):
        dest_alt = [dest_alt]

    n = len(dest_alt)

    for i in range(n):
        if isinstance(dest_alt[i], list):
            dest_alt[i] = dest_alt[i][0]

        vert_dist.append(dest_alt[i] - pos_alt)

    horizontal_dis = compute_great_circle_dist(pos, dest)

    vert_dist_temp = [x ** 2 for x in vert_dist]
    horizontal_dis_temp = [x ** 2 for x in horizontal_dis]
    distance_temp = [sum(x) for x in zip(vert_dist_temp, horizontal_dis_temp)]
    distance = [sqrt(x) for x in distance_temp]
    return distance

def compute_great_circle_dist(pos, dest):
    """
    Calculates the great-circle distance between two points on the Earth's surface.

    Args:
        pos (dict): Start position (latitude, longitude).
        dest (dict): Destination position (latitude, longitude).

    Returns:
        list: List of distances (meters) for each destination.
    """
    if isinstance(pos['latitude'], list):
        pos_lat = pos['latitude'][-1]
        pos_lon = pos['longitude'][-1]
    else:
        pos_lat = pos['latitude']
        pos_lon = pos['longitude']

    dest_lat = dest['latitude']
    dest_lon = dest['longitude']

    if isinstance(dest_lat, float):
        dest_lat = [dest_lat]
        dest_lon = [dest_lon]

    n = len(dest_lat)

    EARTH_RADIUS = 6378137

    distance = []
    for i in range(n):
        if isinstance(dest_lat[i], list):
            dest_lat[i] = dest_lat[i][0]
            dest_lon[i] = dest_lon[i][0]

        dlon = dest_lon[i] - pos_lon
        dlat = dest_lat[i] - pos_lat

        a = sin(dlat / 2) ** 2 + cos(pos_lat) * cos(dest_lat[i]) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance.append(EARTH_RADIUS * c)

    return distance

#not used
def compute_bearing(pos, dest):
    """
    Calculates the initial bearing (azimuth) between two geographic points.

    Args:
        pos (dict): Start position (latitude, longitude).
        dest (dict): Destination position (latitude, longitude).

    Returns:
        list: List of bearings (radians) for each destination.
    """
    pos_lat = pos['latitude']
    pos_lon = pos['longitude']
    dest_lat = dest['latitude']
    dest_lon = dest['longitude']

    if isinstance(dest_lat, float):
        dest_lat = [dest_lat]
        dest_lon = [dest_lon]

    n = len(dest_lat)
    init_course = []
    for i in range(n):
        if isinstance(dest_lat[i], list):
            dest_lat[i] = dest_lat[i][0]
            dest_lon[i] = dest_lon[i][0]

        y = sin(dest_lon[i] - pos_lon) * cos(dest_lat[i])
        x = cos(pos_lat) * sin(dest_lat[i]) - sin(pos_lat) * cos(dest_lat[i]) * cos(dest_lon[i] - pos_lon)

        course = atan2(y, x)
        init_course.append(course + (2 * pi) % (2 * pi))
    return init_course


def find_min_index(list):
    """
    Returns the index of the minimum value in a list.

    Args:
        list (list): List of numeric values.

    Returns:
        int: Index of the minimum value.
    """
    minimum_index = min(range(len(list)), key=lambda i: list[i])
    return minimum_index

def lineXline(pA, pB):
    """
    Calculates the intersection point between two lines in 3D space.

    Args:
        pA (np.ndarray): Points defining the first line (2x3).
        pB (np.ndarray): Points defining the second line (2x3).

    Returns:
        np.ndarray: Coordinates of the intersection point (3x1).
    """

    SI = pA - pB
    NI = np.divide(SI, np.sqrt(np.sum(np.power(SI, 2), axis=0)))
    nX = NI[:, 0]
    nY = NI[:, 1]
    nZ = NI[:, 2]
    SXX = np.sum(np.power(nX, 2) - 1)
    SYY = np.sum(np.power(nY, 2) - 1)
    SZZ = np.sum(np.power(nZ, 2) - 1)
    SXY = np.sum(np.multiply(nX, nY))
    SXZ = np.sum(np.multiply(nX, nZ))
    SYZ = np.sum(np.multiply(nY, nZ))
    S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])
    CX = np.sum(np.multiply(pA[:, 0], np.power(nX, 2) - 1) + np.multiply(pA[:, 1], np.multiply(nX, nY)) + np.multiply(pA[:, 2], np.multiply(nX, nZ)))
    CY = np.sum(np.multiply(pA[:, 0], np.multiply(nX, nY)) + np.multiply(pA[:, 1], np.power(nY, 2) - 1) + np.multiply(pA[:, 2], np.multiply(nY, nZ)))
    CZ = np.sum(np.multiply(pA[:, 0], np.multiply(nX, nZ)) + np.multiply(pA[:, 1], np.multiply(nY, nZ)) + np.multiply(pA[:, 2], np.power(nZ, 2) - 1))
    C = np.array([[CX], [CY], [CZ]])
    P_intersect = np.linalg.solve(S, C)

    return P_intersect


def gotoWaypoint(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, Uidx, params, UAV_data, current_wp_idx):
    """
    Guides a UAV towards its next waypoint while considering flight dynamics, energy constraints, and collision avoidance.
    This function computes candidate trajectories for a UAV based on its current flight mode (glide or engine), 
    evaluates each candidate for safety (collision avoidance with other UAVs), energy consumption, and distance to the goal, 
    and selects the optimal trajectory. The UAV's state and flight conditions are updated accordingly.
    Args:
        FLT_track (list of dict): Flight track history for all UAVs, containing lists of latitude, longitude, altitude, etc.
        FLT_conditions (list of dict): Current flight conditions for all UAVs (airspeed, flight mode, etc.).
        GOAL_WPs (dict): Dictionary of goal waypoints with keys 'latitude', 'longitude', and optionally 'altitude'.
        nUAVs (int): Total number of UAVs in the simulation.
        Uidx (int): Index of the UAV to update.
        params (dict): Simulation and UAV parameters (altitude bounds, time step, safe distance, etc.).
        UAV_data (dict): UAV-specific data (performance, limits, etc.).
        current_wp_idx (int): Index of the current target waypoint for the UAV.
    Returns:
        tuple: Updated (FLT_track, FLT_conditions, current_wp_idx)
            - FLT_track (list of dict): Updated flight track for all UAVs.
            - FLT_conditions (list of dict): Updated flight conditions for all UAVs.
            - current_wp_idx (int): Updated index of the current waypoint for the UAV.
    Notes:
        - The function assumes several helper functions are available, such as compute_distance, get_sink_rate, 
          get_power_consumption, get_destination_from_range_and_bearing, geographic_to_cartesian, cartesian_to_geographic, 
          lineXline, and find_min_index.
        - The function handles both 'glide' and 'engine' flight modes.
        - Collision avoidance is performed by predicting the future positions of other UAVs and checking for conflicts.
        - The function normalizes and combines multiple cost criteria to select the best candidate trajectory.
    """
    
    # Definition
    VO_flag = list()
    PO_flag = list()
    PB_temp = dict()
    PB_temp['latitude'] = []
    PB_temp['longitude'] = []
    PB_temp['altitude'] = []

    
    candidate_sol = dict()
    candidate_sol['latitude'] = []
    candidate_sol['longitude'] = []
    candidate_sol['altitude'] = []
    candidate_sol['bearing'] = []
    candidate_sol['battery_capacity'] = []
    candidate_sol['flight_mode'] = []
    candidate_sol['airspeed'] = []
    candidate_sol['flight_path_angle'] = []
    
    # Initialize flight conditions for the UAV
    LBz = params['altitude_lower_bound']
    UBz = params['altitude_upper_bound']
    Tsim_current = params['current_simulation_time']
    t_step = params['time_step']
    h_step = params['bearing_step']
    v_step = params['speed_step']
    max_turn_rate = UAV_data['max_turn_rate']
    min_velocity = UAV_data['min_airspeed']
    max_velocity = UAV_data['max_airspeed']
    safe_dist = params['safe_distance']
    HorizonLength = params['horizon_length']
    ObstacleUAVs = np.concatenate([np.arange(Uidx), np.arange(Uidx+1, nUAVs)]).tolist()

    # Current position and target waypoint
    current_pos = {
        'latitude': FLT_track[Uidx]['latitude'][-1],
        'longitude': FLT_track[Uidx]['longitude'][-1],
        'altitude': FLT_track[Uidx]['altitude'][-1]
    }
    target_wp = {
        'latitude': GOAL_WPs['latitude'][current_wp_idx],
        'longitude': GOAL_WPs['longitude'][current_wp_idx],
        'altitude': FLT_track[Uidx]['altitude'][-1]  # Keep same altitude for now
    }
    
    # Check if we've reached the current waypoint
    distance_to_wp = compute_distance(current_pos, target_wp)[0]
    next_step_distance = FLT_conditions[Uidx]['airspeed'] * params['time_step']
    if distance_to_wp < params.get('waypoint_threshold', 10.0) or next_step_distance >= distance_to_wp:
        current_wp_idx += 1
        if current_wp_idx >= len(GOAL_WPs['latitude']):
            current_wp_idx = len(GOAL_WPs['latitude']) - 1  # Stay at the last waypoint
        return FLT_track, FLT_conditions, current_wp_idx
    
    # Initialisation du calculateur de trajectoire
    #calculator = TrajectoryCalculator(params, UAV_data)
    
    # Obtention des données de vol actuelles
    FLT_data = get_current_flight_data(FLT_track, FLT_conditions, nUAVs)
    
    # Calcul des trajectoires possibles
    #mode = FLT_data[Uidx]['flight_mode']
    #candidate_sol = calculator.calculate_trajectories(FLT_data, Uidx, mode)

    
    
    Hr = np.linspace(FLT_data[Uidx]['bearing'], FLT_data[Uidx]['bearing'] + max_turn_rate, h_step)
    Hl = np.linspace(FLT_data[Uidx]['bearing'] - max_turn_rate, FLT_data[Uidx]['bearing'], h_step)
    H = np.hstack([Hl, Hr[1:]]).tolist()

    Vr = np.linspace(FLT_data[Uidx]['airspeed'], max_velocity, v_step)
    Vl = np.linspace(min_velocity, FLT_data[Uidx]['airspeed'], v_step)
    V = np.hstack([Vl, Vr[1:]]).tolist()

    if FLT_data[Uidx]['flight_mode'] == 'glide':
        for i in range(len(H)):
            for j in range(len(V)):
                FLT_conditions[Uidx]['airspeed'] = V[j]
                FLT_conditions[Uidx]['airspeed_dot'] = 0.0
                dZ = get_sink_rate(UAV_data, FLT_conditions[Uidx])
                pridction_distance = (abs(FLT_data[Uidx]['altitude'] - LBz)/dZ)*V[j]
                TD = V[j] * t_step
                dZ = -dZ * t_step

                REF = dict()
                REF['latitude'] = FLT_data[Uidx]['latitude']
                REF['longitude'] = FLT_data[Uidx]['longitude']
                lat, lon = get_destination_from_range_and_bearing(REF, TD, H[i])
                alt = min(max(FLT_data[Uidx]['altitude']+dZ, LBz), UBz)

                candidate_sol['latitude'].append(lat)
                candidate_sol['longitude'].append(lon)
                candidate_sol['altitude'].append(alt)
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'])
                candidate_sol['flight_mode'].append('glide')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(0.0)

                lat, lon = get_destination_from_range_and_bearing(REF, pridction_distance, H[i])
                PB_temp['latitude'].append(lat)
                PB_temp['longitude'].append(lon)
                PB_temp['altitude'].append(LBz)

    elif FLT_data[Uidx]['flight_mode'] == 'engine':
        for i in range(len(H)):
            for j in range(len(V)):
                FLT_conditions[Uidx]['airspeed'] = V[j]
                FLT_conditions[Uidx]['airspeed_dot'] = 0.0
                dZ = V[j] * t_step
                pridction_distance = (abs(UBz - FLT_data[Uidx]['altitude'])/dZ)*V[j]
                pwr = get_power_consumption(UAV_data, FLT_conditions[Uidx])
                power_consumption = pwr * (t_step / 3600)

                REF = dict()
                REF['latitude'] = FLT_data[Uidx]['latitude']
                REF['longitude'] = FLT_data[Uidx]['longitude']
                lat, lon = get_destination_from_range_and_bearing(REF, dZ, H[i])
                alt = min(max(FLT_data[Uidx]['altitude']+dZ, LBz), UBz)

                candidate_sol['latitude'].append(lat)
                candidate_sol['longitude'].append(lon)
                candidate_sol['altitude'].append(alt)
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity']-power_consumption)
                candidate_sol['flight_mode'].append('engine')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(0.0)

                lat, lon = get_destination_from_range_and_bearing(REF, pridction_distance, H[i])
                PB_temp['latitude'].append(lat)
                PB_temp['longitude'].append(lon)
                PB_temp['altitude'].append(UBz)
                
    
    # Check for obstacles and compute safety constraints
    Dist2Horizon = dict()
    Dist2Horizon['latitude'] = []
    Dist2Horizon['longitude'] = []
    Dist2Horizon['altitude'] = []

    for o in range(len(ObstacleUAVs)):
        u = ObstacleUAVs[o]
        if FLT_data[u]['flight_mode'] == 'glide':
            dZ = get_sink_rate(UAV_data, FLT_conditions[u])*t_step
            pridction_distance = (abs(FLT_data[u]['altitude'] - LBz) / dZ)*FLT_data[u]['airspeed']

        elif FLT_data[u]['flight_mode'] == 'engine':
            dZ = FLT_data[u]['airspeed']*t_step
            pridction_distance = (abs(UBz - FLT_data[u]['altitude']) / dZ)*FLT_data[u]['airspeed']

        REF = dict()
        REF['latitude'] = FLT_data[u]['latitude']
        REF['longitude'] = FLT_data[u]['longitude']
        lat, lon = get_destination_from_range_and_bearing(REF, pridction_distance, FLT_data[u]['bearing'])
        Dist2Horizon['latitude'].append(lat)
        Dist2Horizon['longitude'].append(lon)
        Dist2Horizon['altitude'].append(LBz)

    x0, y0, z0 = geographic_to_cartesian(FLT_data[Uidx]['latitude'], FLT_data[Uidx]['longitude'])

    for i in range(len(H)*len(V)):
        flag1 = True
        flag2 = True
        xx0, yy0, zz0 = geographic_to_cartesian(PB_temp['latitude'][i], PB_temp['longitude'][i])

        for o in range(len(ObstacleUAVs)):
            u = ObstacleUAVs[o]
            pos = dict()
            pos['latitude'] = FLT_data[u]['latitude']
            pos['longitude'] = FLT_data[u]['longitude']
            pos['altitude'] = FLT_data[u]['altitude']
            dest = dict()
            dest['latitude'] = candidate_sol['latitude'][i]
            dest['longitude'] = candidate_sol['longitude'][i]
            dest['altitude'] = candidate_sol['altitude'][i]
            D = compute_distance(pos, dest)[0]
            if D <= HorizonLength:
                flag1 = flag1 and (D >= safe_dist)

            x1, y1, z1 = geographic_to_cartesian(FLT_data[u]['latitude'], FLT_data[u]['longitude'])
            xx1, yy1, zz1 = geographic_to_cartesian(Dist2Horizon['latitude'][o], Dist2Horizon['longitude'][o])

            pA = np.array(([[x0, y0, z0], [x1, y1, z1]]))
            pB = np.array(([[xx0, yy0, zz0], [xx1, yy1, zz1]]))
            P_intersect = lineXline(pA, pB)

            if not(all(np.isinf(P_intersect))):
                lat, lon, alt = cartesian_to_geographic(P_intersect[0], P_intersect[1], P_intersect[2])

                pos = dict()
                pos['latitude'] = FLT_data[Uidx]['latitude']
                pos['longitude'] = FLT_data[Uidx]['longitude']
                pos['altitude'] = FLT_data[Uidx]['altitude']
                dest = dict()
                dest['latitude'] = lat
                dest['longitude'] = lon
                dest['altitude'] = Dist2Horizon['altitude'][o]  # TO BE CHECKED (SHOULD BE: alt)
                D0 = compute_distance(pos, dest)[0]

                pos['latitude'] = candidate_sol['latitude'][i]
                pos['longitude'] = candidate_sol['longitude'][i]
                pos['altitude'] = candidate_sol['altitude'][i]
                dest['latitude'] = lat
                dest['longitude'] = lon
                dest['altitude'] = Dist2Horizon['altitude'][o]  # TO BE CHECKED (SHOULD BE: alt)
                D1 = compute_distance(pos, dest)[0]

                pos['latitude'] = FLT_data[u]['latitude']
                pos['longitude'] = FLT_data[u]['longitude']
                pos['altitude'] = FLT_data[u]['altitude']
                dest['latitude'] = lat
                dest['longitude'] = lon
                dest['altitude'] = Dist2Horizon['altitude'][o]  # TO BE CHECKED (SHOULD BE: alt)
                D2 = compute_distance(pos, dest)[0]

                t1 = Tsim_current + (D1 / candidate_sol['airspeed'][i])
                t2 = Tsim_current + (D2 / FLT_data[u]['airspeed'])
                DT = abs(t1 - t2)*candidate_sol['airspeed'][i]

                flag2 = flag2 and (D0 >= HorizonLength or (D0 < HorizonLength and DT >= safe_dist))

        VO_flag.append(flag1)
        PO_flag.append(flag2)

    C_safety = []
    C_distance = []
    C_energy = []
    C_sink = []

    for i in range(len(H)*len(V)):
        pos = dict()
        pos['latitude'] = candidate_sol['latitude'][i]
        pos['longitude'] = candidate_sol['longitude'][i]
        pos['altitude'] = candidate_sol['altitude'][i]
        dest = dict()
        dest['latitude'] = GOAL_WPs['latitude'][current_wp_idx]  
        dest['longitude'] = GOAL_WPs['longitude'][current_wp_idx]  
        dest['altitude'] = candidate_sol['altitude'][i]
        D0 = compute_distance(pos, dest)[0]

        C_safety.append(-float(VO_flag[i] and PO_flag[i]))
        C_distance.append(D0)
        C_energy.append((FLT_data[Uidx]['battery_capacity'] - candidate_sol['battery_capacity'][i]) / FLT_data[Uidx]['battery_capacity'])
        C_sink.append((FLT_data[Uidx]['altitude'] - candidate_sol['altitude'][i]) / FLT_data[Uidx]['altitude'])

    C_safety = np.array(C_safety)
    norm_C_safety = np.divide(C_safety, np.linalg.norm(C_safety))
    norm_C_safety[np.isnan(norm_C_safety)] = 0
    C_distance = np.array(C_distance)
    norm_C_distance = np.divide(C_distance, np.linalg.norm(C_distance))
    C_energy = np.array(C_energy)
    norm_C_energy = np.divide(C_energy, np.linalg.norm(C_energy))
    norm_C_energy[np.isnan(norm_C_energy)] = 0
    C_sink = np.array(C_sink)
    norm_C_sink = np.divide(C_sink, np.linalg.norm(C_sink))

    idx = find_min_index((norm_C_safety + norm_C_distance + norm_C_energy + norm_C_sink).tolist())

    FLT_track[Uidx]['latitude'].append(candidate_sol['latitude'][idx])
    FLT_track[Uidx]['longitude'].append(candidate_sol['longitude'][idx])
    FLT_track[Uidx]['altitude'].append(candidate_sol['altitude'][idx])
    FLT_track[Uidx]['bearing'].append(candidate_sol['bearing'][idx])
    FLT_track[Uidx]['flight_mode'].append(candidate_sol['flight_mode'][idx])
    FLT_track[Uidx]['battery_capacity'].append(candidate_sol['battery_capacity'][idx])

    FLT_conditions[Uidx]['airspeed'] = candidate_sol['airspeed'][idx]
    FLT_conditions[Uidx]['flight_path_angle'] = candidate_sol['flight_path_angle'][idx]

    return FLT_track, FLT_conditions, current_wp_idx

def gotoWaypointMulti(FLT_track: Dict, FLT_conditions: Dict, GOAL_WPs: Dict, 
                    nUAVs: int, params: Dict, UAV_data: Dict, 
                    current_wp_idx: Dict[int, int]) -> Tuple[Dict, Dict, Dict[int, int]]:
    """
    Gère et contrôle plusieurs UAVs simultanément, en suivant et en mettant à jour leur progression vers leurs waypoints respectifs.

    Args:
        FLT_track (dict): Historique des positions et états des UAVs.
        FLT_conditions (dict): Conditions de vol actuelles des UAVs.
        GOAL_WPs (dict): Waypoints cibles (latitude, longitude).
        nUAVs (int): Nombre total de UAVs.
        params (dict): Paramètres de simulation.
        UAV_data (dict): Paramètres physiques du UAV (identiques pour tous ou par UAV).
        current_wp_indices (dict): Dictionnaire {Uidx: index_wp_courant} pour chaque UAV.

    Returns:
        tuple: (FLT_track, FLT_conditions, current_wp_indices) mis à jour pour tous les UAVs traités.
    """
    for Uidx in range(nUAVs):
        # Récupérer l'index du waypoint courant pour ce UAV
        wp_idx = current_wp_idx.get(Uidx, 0)
        # Appeler la logique existante pour un seul UAV
        FLT_track, FLT_conditions, new_wp_idx = gotoWaypoint(
            FLT_track, FLT_conditions, GOAL_WPs, nUAVs, Uidx, params, UAV_data, wp_idx
        )
        # Mettre à jour l'index du waypoint pour ce UAV
        current_wp_idx[Uidx] = new_wp_idx
    return FLT_track, FLT_conditions, current_wp_idx