from math import cos, sin, pi, sqrt, atan, atan2
import numpy as np
import warnings
from compute import (
    cartesian_to_geographic,
    compute_distance,
    geographic_to_cartesian,
    get_current_flight_data,
    get_power_consumption,
    get_sink_rate,
    get_destination_from_range_and_bearing_cartesian
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
        GOAL_WPs (dict): Target waypoints for the UAV, structured as a dictionary with keys 'latitude', 'longitude', and 'altitude'.
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
    PB_temp['X'] = []
    PB_temp['Y'] = []
    PB_temp['Z'] = []

    
    candidate_sol = dict()
    candidate_sol['X'] = []
    candidate_sol['Y'] = []
    candidate_sol['Z'] = []
    candidate_sol['bearing'] = []
    candidate_sol['battery_capacity'] = []
    candidate_sol['flight_mode'] = []
    candidate_sol['airspeed'] = []
    candidate_sol['flight_path_angle'] = []
    
    # Initialize flight conditions for the UAV
    LBz = params['Z_lower_bound']
    UBz = params['Z_upper_bound']
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
        'X': FLT_track[Uidx]['X'][-1],
        'Y': FLT_track[Uidx]['Y'][-1],
        'Z': FLT_track[Uidx]['Z'][-1]
    }
    target_wp = {
        'X': GOAL_WPs['X'][current_wp_idx],
        'Y': GOAL_WPs['Y'][current_wp_idx],
        'Z': GOAL_WPs['Z'][current_wp_idx]
    }
    
    # Check if we've reached the current waypoint
    distance_to_wp = compute_distance(current_pos, target_wp)[0]
    next_step_distance = FLT_conditions[Uidx]['airspeed'] * params['time_step']
    if distance_to_wp < params.get('waypoint_threshold', 10.0) or next_step_distance >= distance_to_wp:
        current_wp_idx += 1
        if current_wp_idx >= len(GOAL_WPs['X']):
            current_wp_idx = len(GOAL_WPs['X']) - 1  # Stay at the last waypoint
        return FLT_track, FLT_conditions, current_wp_idx

    # Obtention des données de vol actuelles
    FLT_data = get_current_flight_data(FLT_track, FLT_conditions, nUAVs)
    
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
                pridction_distance = (abs(FLT_data[Uidx]['Z'] - LBz)/dZ)*V[j] if dZ != 0 else V[j] * t_step
                TD = V[j] * t_step
                dZ = -dZ * t_step

                REF = dict()
                REF['X'] = FLT_data[Uidx]['X']
                REF['Y'] = FLT_data[Uidx]['Y']
                x_new, y_new = get_destination_from_range_and_bearing_cartesian(REF, TD, H[i])
                alt = min(max(FLT_data[Uidx]['Z']+dZ, LBz), UBz)

                candidate_sol['X'].append(x_new)
                candidate_sol['Y'].append(y_new)
                candidate_sol['Z'].append(alt)
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'])
                candidate_sol['flight_mode'].append('glide')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(0.0)
                

                x_pred, y_pred = get_destination_from_range_and_bearing_cartesian(REF, pridction_distance, H[i])
                PB_temp['X'].append(x_pred)
                PB_temp['Y'].append(y_pred)
                PB_temp['Z'].append(LBz)

    elif FLT_data[Uidx]['flight_mode'] == 'engine':
        for i in range(len(H)):
            for j in range(len(V)):
                FLT_conditions[Uidx]['airspeed'] = V[j]
                FLT_conditions[Uidx]['airspeed_dot'] = 0.0
                dZ = V[j] * t_step
                pridction_distance = (abs(UBz - FLT_data[Uidx]['Z'])/dZ)*V[j] if dZ != 0 else V[j] * t_step
                pwr = get_power_consumption(UAV_data, FLT_conditions[Uidx])
                power_consumption = pwr * (t_step / 3600)

                REF = dict()
                REF['X'] = FLT_data[Uidx]['X']
                REF['Y'] = FLT_data[Uidx]['Y']
                x_new, y_new = get_destination_from_range_and_bearing_cartesian(REF, dZ, H[i])
                alt = min(max(FLT_data[Uidx]['Z']+dZ, LBz), UBz)

                candidate_sol['X'].append(x_new)
                candidate_sol['Y'].append(y_new)
                candidate_sol['Z'].append(alt)
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity']-power_consumption)
                candidate_sol['flight_mode'].append('engine')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(0.0)

                x_pred, y_pred = get_destination_from_range_and_bearing_cartesian(REF, pridction_distance, H[i])
                PB_temp['X'].append(x_pred)
                PB_temp['Y'].append(y_pred)
                PB_temp['Z'].append(UBz)
    
    # Check for obstacles and compute safety constraints
    Dist2Horizon = dict()
    Dist2Horizon['X'] = []
    Dist2Horizon['Y'] = []
    Dist2Horizon['Z'] = []

    for o in range(len(ObstacleUAVs)):
        u = ObstacleUAVs[o]
        if FLT_data[u]['flight_mode'] == 'glide':
            dZ = get_sink_rate(UAV_data, FLT_conditions[u])*t_step
            pridction_distance = (abs(FLT_data[u]['Z'] - LBz) / dZ)*FLT_data[u]['airspeed'] if dZ != 0 else FLT_data[u]['airspeed'] * t_step

        elif FLT_data[u]['flight_mode'] == 'engine':
            dZ = FLT_data[u]['airspeed']*t_step
            pridction_distance = (abs(UBz - FLT_data[u]['Z']) / dZ)*FLT_data[u]['airspeed'] if dZ != 0 else FLT_data[u]['airspeed'] * t_step

        REF = dict()
        REF['X'] = FLT_data[u]['X']
        REF['Y'] = FLT_data[u]['Y']
        x_obs, y_obs = get_destination_from_range_and_bearing_cartesian(REF, pridction_distance, FLT_data[u]['bearing'])
        Dist2Horizon['X'].append(x_obs)
        Dist2Horizon['Y'].append(y_obs)
        Dist2Horizon['Z'].append(LBz)

    x0, y0, z0 = FLT_data[Uidx]['X'], FLT_data[Uidx]['Y'], FLT_data[Uidx]['Z']

    for i in range(len(H)*len(V)):
        flag1 = True
        flag2 = True
        xx0, yy0, zz0 = PB_temp['X'][i], PB_temp['Y'][i], PB_temp['Z'][i]

        for o in range(len(ObstacleUAVs)):
            u = ObstacleUAVs[o]
            pos = dict()
            pos['X'] = FLT_data[u]['X']
            pos['Y'] = FLT_data[u]['Y']
            pos['Z'] = FLT_data[u]['Z']
            dest = dict()
            dest['X'] = candidate_sol['X'][i]
            dest['Y'] = candidate_sol['Y'][i]
            dest['Z'] = candidate_sol['Z'][i]
            D = compute_distance(pos, dest)[0]
            if D <= HorizonLength:
                flag1 = flag1 and (D >= safe_dist)

            x1, y1, z1 = FLT_data[u]['X'], FLT_data[u]['Y'], FLT_data[u]['Z']
            xx1, yy1, zz1 = Dist2Horizon['X'][o], Dist2Horizon['Y'][o], Dist2Horizon['Z'][o]

            pA = np.array(([[x0, y0, z0], [x1, y1, z1]]))
            pB = np.array(([[xx0, yy0, zz0], [xx1, yy1, zz1]]))
            P_intersect = lineXline(pA, pB)

            if not(all(np.isinf(P_intersect))):
                x_int, y_int, z_int = P_intersect[0], P_intersect[1], P_intersect[2]

                pos = dict()
                pos['X'] = FLT_data[Uidx]['X']
                pos['Y'] = FLT_data[Uidx]['Y']
                pos['Z'] = FLT_data[Uidx]['Z']
                dest = dict()
                dest['X'] = x_int
                dest['Y'] = y_int
                dest['Z'] = z_int
                D0 = compute_distance(pos, dest)[0]

                pos['X'] = candidate_sol['X'][i]
                pos['Y'] = candidate_sol['Y'][i]
                pos['Z'] = candidate_sol['Z'][i]
                dest['X'] = x_int
                dest['Y'] = y_int
                dest['Z'] = z_int
                D1 = compute_distance(pos, dest)[0]

                pos['X'] = FLT_data[u]['X']
                pos['Y'] = FLT_data[u]['Y']
                pos['Z'] = FLT_data[u]['Z']
                dest['X'] = x_int
                dest['Y'] = y_int
                dest['Z'] = z_int
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
        pos['X'] = candidate_sol['X'][i]
        pos['Y'] = candidate_sol['Y'][i]
        pos['Z'] = candidate_sol['Z'][i]
        dest = dict()
        dest['X'] = GOAL_WPs['X'][current_wp_idx]  
        dest['Y'] = GOAL_WPs['Y'][current_wp_idx]  
        dest['Z'] = GOAL_WPs['Z'][current_wp_idx]
        D0 = compute_distance(pos, dest)[0]

        C_safety.append(-float(VO_flag[i] and PO_flag[i]))
        C_distance.append(D0)
        C_energy.append((FLT_data[Uidx]['battery_capacity'] - candidate_sol['battery_capacity'][i]) / FLT_data[Uidx]['battery_capacity'])
        C_sink.append((FLT_data[Uidx]['Z'] - candidate_sol['Z'][i]) / FLT_data[Uidx]['Z'])

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

    FLT_track[Uidx]['X'].append(candidate_sol['X'][idx])
    FLT_track[Uidx]['Y'].append(candidate_sol['Y'][idx])
    FLT_track[Uidx]['Z'].append(candidate_sol['Z'][idx])
    FLT_track[Uidx]['bearing'].append(candidate_sol['bearing'][idx])

    if FLT_track[Uidx]['Z'][-1] <= LBz:
        FLT_track[Uidx]['flight_mode'].append('engine')
    elif FLT_track[Uidx]['Z'][-1] >= UBz:
        FLT_track[Uidx]['flight_mode'].append('glide')
    else:
        FLT_track[Uidx]['flight_mode'].append(candidate_sol['flight_mode'][idx])
    FLT_track[Uidx]['battery_capacity'].append(candidate_sol['battery_capacity'][idx])

    FLT_conditions[Uidx]['airspeed'] = candidate_sol['airspeed'][idx]
    FLT_conditions[Uidx]['flight_path_angle'] = candidate_sol['flight_path_angle'][idx]

    return FLT_track, FLT_conditions, current_wp_idx

def gotoWaypointMulti(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data, current_wp_indices):
    """
    Gère et contrôle plusieurs UAVs simultanément, en suivant et en mettant à jour leur progression vers leurs waypoints respectifs.

    Args:
        FLT_track (dict): Historique des positions et états des UAVs.
        FLT_conditions (dict): Conditions de vol actuelles des UAVs.
        GOAL_WPs (dict): Dictionnaire {Uidx: [ {'X', 'Y', 'Z'}] } pour chaque UAV.
        nUAVs (int): Nombre total de UAVs.
        params (dict): Paramètres de simulation.
        UAV_data (dict): Paramètres physiques du UAV (identiques pour tous ou par UAV).
        current_wp_indices (dict): Dictionnaire {Uidx: index_wp_courant} pour chaque UAV.

    Returns:
        tuple: (FLT_track, FLT_conditions, current_wp_indices) mis à jour pour tous les UAVs traités.
    """
    for Uidx in range(nUAVs):
        # Récupérer l'index du waypoint courant pour ce UAV
        wp_idx = current_wp_indices[Uidx]

        FLT_track, FLT_conditions, new_wp_idx = gotoWaypoint(
            FLT_track, FLT_conditions, GOAL_WPs[Uidx], nUAVs, Uidx, params, UAV_data, wp_idx
        )
        # Mettre à jour l'index du waypoint pour ce UAV
        current_wp_indices[Uidx] = new_wp_idx
    return FLT_track, FLT_conditions, current_wp_indices