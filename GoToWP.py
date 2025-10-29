import numpy as np
import warnings
from compute import (
    calculate_optimal_climb_angle,
    check_segment_obstacle_collision,
    compute_bearing_cartesian,
    compute_distance_cartesian,
    convert_cylindrical_obstacles_to_polygons,
    decision_making,
    find_nearest_waypoint,
    get_current_flight_data,
    get_power_consumption,
    get_sink_rate,
    get_destination_from_range_and_bearing_cartesian,
    is_point_in_obstacle
)
from thermal import detect_thermal_at_position

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
    OB_flag = list()
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
    obstacles = params.get('obstacles', [])
    
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
    distance_to_wp = np.sqrt((current_pos['X'] - target_wp['X'])**2 + (current_pos['Y'] - target_wp['Y'])**2)
    print(f"Distance to waypoint {current_wp_idx} for UAV {Uidx}: {distance_to_wp:.2f} m")
    next_step_distance = FLT_conditions[Uidx]['airspeed'] * params['time_step']
    
    # Filtrer les obstacles - ignorer l'obstacle si c'est le drone qui fait l'évaluation
    obstacles_to_check = obstacles
    if FLT_track[Uidx].get('in_evaluation', False) and FLT_track[Uidx].get('current_thermal_id') is not None:
        thermal_id = FLT_track[Uidx]['current_thermal_id']
        obstacles_to_check = [obs for obs in obstacles if obs.get('thermal_id') != thermal_id]
    
    if distance_to_wp <= 10 or next_step_distance >= distance_to_wp or any(is_point_in_obstacle(target_wp, obstacle) for obstacle in obstacles_to_check):
        current_wp_idx += 1
        if current_wp_idx >= len(GOAL_WPs['X']):
            current_wp_idx = len(GOAL_WPs['X'])
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
                dZ_sink = -dZ * t_step

                REF = dict()
                REF['X'] = FLT_data[Uidx]['X']
                REF['Y'] = FLT_data[Uidx]['Y']
                x_new, y_new = get_destination_from_range_and_bearing_cartesian(REF, TD, H[i])

                alt = min(max(FLT_data[Uidx]['Z'] + dZ_sink, LBz), UBz)

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
        # Trois possibilités avec le moteur: monter, voler à niveau constant, descendre de manière contrôlée
        
        # Définir les angles de trajectoire pour chaque mode
        climb_angle = calculate_optimal_climb_angle(UAV_data, FLT_conditions[Uidx])
        level_angle = 0
        descent_angle = np.deg2rad(-5)

        # Pour chaque cap possible
        for i in range(len(H)):
            for j in range(len(V)):
                FLT_conditions[Uidx]['airspeed'] = V[j]
                FLT_conditions[Uidx]['airspeed_dot'] = 0.0
                
                # 1. MONTÉE AVEC MOTEUR
                FLT_conditions[Uidx]['flight_path_angle'] = climb_angle
                
                # Calculer la distance parcourue et le changement d'altitude
                horizontal_distance = V[j] * t_step * np.cos(climb_angle)
                altitude_change = V[j] * t_step * np.sin(climb_angle)
                
                # Calculer la consommation d'énergie pour la montée
                pwr_climb = get_power_consumption(UAV_data, FLT_conditions[Uidx])
                power_consumption_climb = pwr_climb * (t_step / 3600)
                
                # Calculer la nouvelle position
                REF = dict()
                REF['X'] = FLT_data[Uidx]['X']
                REF['Y'] = FLT_data[Uidx]['Y']
                x_new_climb, y_new_climb = get_destination_from_range_and_bearing_cartesian(REF, horizontal_distance, H[i])
                alt_climb = min(max(FLT_data[Uidx]['Z'] + altitude_change, LBz), UBz)
                
                # Ajouter le candidat pour la montée
                candidate_sol['X'].append(x_new_climb)
                candidate_sol['Y'].append(y_new_climb)
                candidate_sol['Z'].append(alt_climb)
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'] - power_consumption_climb)
                candidate_sol['flight_mode'].append('engine')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(climb_angle)
                
                # Prédiction pour collision
                pridction_distance = (abs(UBz - FLT_data[Uidx]['Z']) / altitude_change) * horizontal_distance if abs(altitude_change) > 1e-6 else V[j] * t_step
                x_pred_climb, y_pred_climb = get_destination_from_range_and_bearing_cartesian(REF, pridction_distance, H[i])
                PB_temp['X'].append(x_pred_climb)
                PB_temp['Y'].append(y_pred_climb)
                PB_temp['Z'].append(UBz)
                
                # 2. VOL À NIVEAU CONSTANT AVEC MOTEUR
                FLT_conditions[Uidx]['flight_path_angle'] = level_angle
                
                # Distance parcourue horizontalement (égale à la vitesse * temps)
                horizontal_distance = V[j] * t_step
                
                # Calculer la consommation d'énergie pour vol à niveau
                pwr_level = get_power_consumption(UAV_data, FLT_conditions[Uidx])
                power_consumption_level = pwr_level * (t_step / 3600)
                
                # Calculer la nouvelle position (altitude inchangée)
                x_new_level, y_new_level = get_destination_from_range_and_bearing_cartesian(REF, horizontal_distance, H[i])
                
                # Ajouter le candidat pour le vol à niveau
                candidate_sol['X'].append(x_new_level)
                candidate_sol['Y'].append(y_new_level)
                candidate_sol['Z'].append(FLT_data[Uidx]['Z'])  # Altitude inchangée
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'] - power_consumption_level)
                candidate_sol['flight_mode'].append('engine')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(level_angle)
                
                # Prédiction pour collision (distance horizontale directe vers la destination)
                target_distance = np.sqrt((GOAL_WPs['X'][current_wp_idx] - FLT_data[Uidx]['X'])**2 + 
                                         (GOAL_WPs['Y'][current_wp_idx] - FLT_data[Uidx]['Y'])**2)
                x_pred_level, y_pred_level = get_destination_from_range_and_bearing_cartesian(REF, target_distance, H[i])
                PB_temp['X'].append(x_pred_level)
                PB_temp['Y'].append(y_pred_level)
                PB_temp['Z'].append(FLT_data[Uidx]['Z'])
                
                # 3. DESCENTE CONTRÔLÉE AVEC MOTEUR
                FLT_conditions[Uidx]['flight_path_angle'] = descent_angle
                
                # Calculer la distance parcourue et le changement d'altitude
                horizontal_distance = V[j] * t_step * np.cos(descent_angle)
                altitude_change = V[j] * t_step * np.sin(descent_angle)  # Négatif car descente
                
                # Calculer la consommation d'énergie pour la descente contrôlée
                pwr_descent = get_power_consumption(UAV_data, FLT_conditions[Uidx])
                power_consumption_descent = pwr_descent * (t_step / 3600)
                
                # Calculer la nouvelle position
                x_new_descent, y_new_descent = get_destination_from_range_and_bearing_cartesian(REF, horizontal_distance, H[i])
                alt_descent = min(max(FLT_data[Uidx]['Z'] + altitude_change, LBz), UBz)
                
                # Ajouter le candidat pour la descente
                candidate_sol['X'].append(x_new_descent)
                candidate_sol['Y'].append(y_new_descent)
                candidate_sol['Z'].append(alt_descent)
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'] - power_consumption_descent)
                candidate_sol['flight_mode'].append('engine')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(descent_angle)
                
                # Prédiction pour collision
                pridction_distance = (abs(FLT_data[Uidx]['Z'] - LBz) / abs(altitude_change)) * horizontal_distance if abs(altitude_change) > 1e-6 else V[j] * t_step
                x_pred_descent, y_pred_descent = get_destination_from_range_and_bearing_cartesian(REF, pridction_distance, H[i])
                PB_temp['X'].append(x_pred_descent)
                PB_temp['Y'].append(y_pred_descent)
                PB_temp['Z'].append(LBz)


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

    for i in range(len(candidate_sol['X'])):
        flag1 = True
        flag2 = True
        flag3 = True
        xx0, yy0, zz0 = PB_temp['X'][i], PB_temp['Y'][i], PB_temp['Z'][i]
        
        current_point = {
            'X': x0,
            'Y': y0,
            'Z': z0
        }
        
        candidate_point = {
            'X': candidate_sol['X'][i],
            'Y': candidate_sol['Y'][i],
            'Z': candidate_sol['Z'][i]
        }
        
         # Vérifier les collisions avec les obstacles
        for obstacle in obstacles:
            # Vérifier si le point de destination est dans un obstacle
            if is_point_in_obstacle(candidate_point, obstacle):
                flag3 = False
                break
                
            # Vérifier si le segment traverse un obstacle
            if check_segment_obstacle_collision(current_point, candidate_point, obstacle):
                flag3 = False
                break

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
            D = compute_distance_cartesian(pos, dest)[0]
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
                D0 = compute_distance_cartesian(pos, dest)[0]

                pos['X'] = candidate_sol['X'][i]
                pos['Y'] = candidate_sol['Y'][i]
                pos['Z'] = candidate_sol['Z'][i]
                dest['X'] = x_int
                dest['Y'] = y_int
                dest['Z'] = z_int
                D1 = compute_distance_cartesian(pos, dest)[0]

                pos['X'] = FLT_data[u]['X']
                pos['Y'] = FLT_data[u]['Y']
                pos['Z'] = FLT_data[u]['Z']
                dest['X'] = x_int
                dest['Y'] = y_int
                dest['Z'] = z_int
                D2 = compute_distance_cartesian(pos, dest)[0]

                t1 = Tsim_current + (D1 / candidate_sol['airspeed'][i])
                t2 = Tsim_current + (D2 / FLT_data[u]['airspeed'])
                DT = abs(t1 - t2)*candidate_sol['airspeed'][i]

                flag2 = flag2 and (D0 >= HorizonLength or (D0 < HorizonLength and DT >= safe_dist))

        VO_flag.append(flag1)
        PO_flag.append(flag2)
        OB_flag.append(flag3)

    C_safety = []
    C_distance = []
    C_energy = []
    C_sink = []
    C_obstacle = []

    for i in range(len(candidate_sol['X'])):
        pos = dict()
        pos['X'] = candidate_sol['X'][i]
        pos['Y'] = candidate_sol['Y'][i]
        pos['Z'] = candidate_sol['Z'][i]
        dest = dict()
        dest['X'] = GOAL_WPs['X'][current_wp_idx]  
        dest['Y'] = GOAL_WPs['Y'][current_wp_idx]  
        dest['Z'] = GOAL_WPs['Z'][current_wp_idx]
        D0 = compute_distance_cartesian(pos, dest)[0]

        C_safety.append(-float(VO_flag[i] and PO_flag[i]))
        C_distance.append(D0)
        C_energy.append((FLT_data[Uidx]['battery_capacity'] - candidate_sol['battery_capacity'][i]) / FLT_data[Uidx]['battery_capacity'])
        C_sink.append((FLT_data[Uidx]['Z'] - candidate_sol['Z'][i]) / FLT_data[Uidx]['Z'])
        C_obstacle.append(-float(OB_flag[i]))  # -1 si collision avec obstacle, 0 sinon
        
    C_safety = np.array(C_safety)
    C_distance = np.array(C_distance)
    C_energy = np.array(C_energy)
    C_sink = np.array(C_sink)
    C_obstacle = np.array(C_obstacle)

    DM = np.column_stack([C_safety, C_distance, C_energy, C_sink, C_obstacle])
    ranked_indices = decision_making(DM)
    idx = ranked_indices[0]  # Meilleur candidat selon le classement

    FLT_track[Uidx]['X'].append(candidate_sol['X'][idx])
    FLT_track[Uidx]['Y'].append(candidate_sol['Y'][idx])
    FLT_track[Uidx]['Z'].append(candidate_sol['Z'][idx])
    FLT_track[Uidx]['bearing'].append(candidate_sol['bearing'][idx])
    
    candidate_mode = candidate_sol['flight_mode'][idx]

    # Si le mode n'est pas soaring, on vérifie les conditions de vol
    # Si engine et altitude plus petit que working floor ou si glide et altitude plus petit que LBz alors engine else glide
    if (candidate_mode == 'engine' and FLT_track[Uidx]['Z'][-1] <= params['working_floor'] and FLT_track[Uidx]['current_thermal_id'] is None) or \
       (candidate_mode == 'glide' and FLT_track[Uidx]['Z'][-1] <= LBz and FLT_track[Uidx]['current_thermal_id'] is None):
        FLT_track[Uidx]['flight_mode'].append('engine')
    else:
        FLT_track[Uidx]['flight_mode'].append('glide')

    FLT_track[Uidx]['battery_capacity'].append(candidate_sol['battery_capacity'][idx])

    FLT_conditions[Uidx]['airspeed'] = candidate_sol['airspeed'][idx]
    FLT_conditions[Uidx]['flight_path_angle'] = candidate_sol['flight_path_angle'][idx]
    
    return FLT_track, FLT_conditions, current_wp_idx

def soarThermal(FLT_track, FLT_conditions, SOAR_WPs, nUAVs, Uidx, params, UAV_data, wp_idx, thermal_map):
    """
    Guide un UAV le long d'une trajectoire de soaring dans un thermique.
    La vérification des conditions de sortie est gérée par gotoWaypointMulti.
    
    Args:
        FLT_track (dict): Historique de vol pour tous les UAVs
        FLT_conditions (dict): Conditions de vol actuelles pour tous les UAVs
        SOAR_WPs (dict): Waypoints de soaring générés {'X': [], 'Y': [], 'Z': [], 'bearing': [], 'flight_path_angle': [], 'bank_angle': []}
        nUAVs (int): Nombre total d'UAVs
        Uidx (int): Index de l'UAV actuel
        params (dict): Paramètres de simulation
        UAV_data (dict): Données du UAV
        wp_idx (int): Index du waypoint actuel dans la trajectoire de soaring
        thermal_map (ThermalMap): Carte des thermiques
        
    Returns:
        tuple: (FLT_track, FLT_conditions, new_wp_idx, updated_SOAR_WPs)
    """
    
    # Obtenir les données de vol actuelles
    FLT_data = get_current_flight_data(FLT_track, FLT_conditions, nUAVs)
    
    # Position actuelle
    current_pos = {
        'X': FLT_data[Uidx]['X'],
        'Y': FLT_data[Uidx]['Y'],
        'Z': FLT_data[Uidx]['Z']
    }
    
    # Vérifier s'il y a des waypoints de soaring disponibles
    if not SOAR_WPs or len(SOAR_WPs['X']) == 0:
        return FLT_track, FLT_conditions, wp_idx, SOAR_WPs
    
    # Vérifier si nous avons atteint la fin de la trajectoire
    if wp_idx >= len(SOAR_WPs['X']):
        print(f"UAV {Uidx}: Fin de trajectoire de soaring atteinte (wp_idx={wp_idx}, total_wp={len(SOAR_WPs['X'])})")
        return FLT_track, FLT_conditions, wp_idx, SOAR_WPs
    
    # Obtenir le waypoint cible actuel
    target_wp = {
        'X': SOAR_WPs['X'][wp_idx],
        'Y': SOAR_WPs['Y'][wp_idx]
    }
    
    # Obtenir les angles cibles (en radians)
    target_bearing = SOAR_WPs['bearing'][wp_idx]  # En radians
    target_bank_angle_rad = SOAR_WPs['bank_angle'][wp_idx]  # En radians
    
    # Calculer la distance au waypoint cible
    distance_to_wp = np.sqrt((current_pos['X'] - target_wp['X'])**2 + 
                            (current_pos['Y'] - target_wp['Y'])**2)
    
    # Distance parcourue par pas de temps
    airspeed = FLT_conditions[Uidx]['airspeed']
    step_distance = airspeed * params['time_step']
    
    # Vérifier si nous avons atteint le waypoint (seuil de proximité)
    proximity_threshold = max(10.0, step_distance * 0.5)
    
    if distance_to_wp <= proximity_threshold:
        # Waypoint atteint, passer au suivant
        wp_idx += 1
        print(f"UAV {Uidx}: Waypoint {wp_idx-1} atteint, passage au waypoint {wp_idx}")
        
        # Si nous avons atteint la fin de la trajectoire, retourner
        if wp_idx >= len(SOAR_WPs['X']):
            return FLT_track, FLT_conditions, wp_idx, SOAR_WPs
        
        # Mettre à jour le waypoint cible avec le prochain point
        target_wp = {
            'X': SOAR_WPs['X'][wp_idx],
            'Y': SOAR_WPs['Y'][wp_idx]
        }
        target_bearing = SOAR_WPs['bearing'][wp_idx]
        target_bank_angle_rad = SOAR_WPs['bank_angle'][wp_idx]
    
    # Calculer la prochaine position en suivant la trajectoire de soaring
    bearing_to_target = np.arctan2(target_wp['Y'] - current_pos['Y'], 
                                  target_wp['X'] - current_pos['X'])
    
    # Normaliser le bearing
    bearing_to_target = (bearing_to_target + 2 * np.pi) % (2 * np.pi)
    
    # INITIALISER effective_bearing avec une valeur par défaut
    effective_bearing = bearing_to_target  # Valeur par défaut
    
    # Calcul de l'angle d'inclinaison optimal dynamique
    thermal_id = FLT_data[Uidx]['current_thermal_id']   
    dynamic_flight_path_angle = 0.0  # Valeur par défaut
    z_new = current_pos['Z']  # Valeur par défaut
    
    # Calculer la nouvelle position
    REF = {
        'X': current_pos['X'],
        'Y': current_pos['Y']
    }
    
    x_new, y_new = get_destination_from_range_and_bearing_cartesian(
        REF, step_distance, effective_bearing
    )
    
    if thermal_id is not None:
        active_thermals = thermal_map.get_active_thermals(params['current_simulation_time'])
        if thermal_id in active_thermals:
            thermal = active_thermals[thermal_id]['thermal']
            
            # Calculer les paramètres optimaux
            from compute import calculate_optimal_soaring_parameters
            soaring_params = calculate_optimal_soaring_parameters(UAV_data, thermal, FLT_conditions[Uidx])
            
            # Utiliser les paramètres optimaux (en radians)
            optimal_radius = soaring_params['optimal_radius']
            optimal_speed = soaring_params['optimal_speed']
            optimal_bank_angle_rad = soaring_params['optimal_bank_angle']  # En radians
            
            # Ajustements progressifs
            if abs(airspeed - optimal_speed) > 0.5:
                airspeed = airspeed + 0.1 * (optimal_speed - airspeed)
                FLT_conditions[Uidx]['airspeed'] = airspeed
            
            # Ajustement de l'angle d'inclinaison (en radians)
            current_bank_angle_rad = FLT_conditions[Uidx]['bank_angle']  # En radians
            if abs(current_bank_angle_rad - optimal_bank_angle_rad) > np.radians(1.0):
                new_bank_angle_rad = current_bank_angle_rad + 0.1 * (optimal_bank_angle_rad - current_bank_angle_rad)
                target_bank_angle_rad = new_bank_angle_rad
            else:
                target_bank_angle_rad = optimal_bank_angle_rad
            
            # Limiter optimal_radius pour rester dans le thermique
            max_safe_radius = thermal.radius * 0.8
            target_radius = min(optimal_radius, max_safe_radius)
            
            # Vérifier la distance actuelle au centre
            current_distance_to_center = np.sqrt((current_pos['X'] - thermal.x)**2 + (current_pos['Y'] - thermal.y)**2)
            
            # Stratégie d'ajustement vers optimal_radius
            if current_distance_to_center > target_radius * 1.05:
                optimal_angle = np.arctan2(current_pos['Y'] - thermal.y, current_pos['X'] - thermal.x)
                optimal_x = thermal.x + target_radius * np.cos(optimal_angle)
                optimal_y = thermal.y + target_radius * np.sin(optimal_angle)
                
                bearing_to_optimal = np.arctan2(optimal_y - current_pos['Y'], optimal_x - current_pos['X'])
                effective_bearing = bearing_to_optimal
                print(f"UAV {Uidx}: Convergence vers optimal_radius ({current_distance_to_center:.1f}m -> {target_radius:.1f}m), "
                      f"bank_angle: {np.degrees(target_bank_angle_rad):.1f}°")
            else:
                weight = min(distance_to_wp / 30.0, 1.0)
                effective_bearing = target_bearing * weight + bearing_to_target * (1 - weight)
                print(f"UAV {Uidx}: Maintien à optimal_radius ({current_distance_to_center:.1f}m ≈ {target_radius:.1f}m)")
            
            # Calculer l'effet du thermique à la nouvelle position
            distance_from_center = np.sqrt((x_new - thermal.x)**2 + (y_new - thermal.y)**2)
            thermal_lift_rate = thermal.get_lift_rate(distance_from_center)
            
            # Calculer le sink rate avec l'angle d'inclinaison (en radians)
            test_conditions = FLT_conditions[Uidx].copy()
            test_conditions['airspeed'] = airspeed
            test_conditions['bank_angle'] = target_bank_angle_rad  # En radians
            test_conditions['flight_path_angle'] = 0.0  # Vol en palier pour le calcul initial
            
            sink_rate = get_sink_rate(UAV_data, test_conditions)
            
            # Calculer le taux de montée net avec le lift rate effectif (adaptatif)
            net_climb_rate = thermal_lift_rate - sink_rate
            
            # Calculer l'angle de trajectoire dynamiquement
            # γ = arcsin(taux_montée_net / vitesse)
            if airspeed > 0:
                sin_gamma = net_climb_rate / airspeed
                # Limiter sin_gamma dans [-1, 1] pour éviter les erreurs
                sin_gamma = max(-1.0, min(1.0, sin_gamma))
                dynamic_flight_path_angle = np.arcsin(sin_gamma)
            else:
                dynamic_flight_path_angle = 0.0
            
            # Calculer le changement d'altitude basé sur l'angle de trajectoire dynamique
            altitude_change = airspeed * np.sin(dynamic_flight_path_angle) * params['time_step']
            
            z_new = current_pos['Z'] + altitude_change
            
        else:
            # Thermique non trouvée dans active_thermals, utiliser les valeurs par défaut
            weight = min(distance_to_wp / 30.0, 1.0)
            effective_bearing = target_bearing * weight + bearing_to_target * (1 - weight)
            print(f"UAV {Uidx}: Thermique non trouvée, utilisation bearing par défaut")
    else:
        # Pas de thermique, utiliser les valeurs par défaut
        weight = min(distance_to_wp / 30.0, 1.0)
        effective_bearing = target_bearing * weight + bearing_to_target * (1 - weight)
        print(f"UAV {Uidx}: Pas de thermique ID, utilisation bearing par défaut")
    
    

    # Limiter l'altitude aux bornes définies
    z_new = min(max(z_new, params['Z_lower_bound']), params['Z_upper_bound'])
    
    # Mettre à jour la trajectoire de vol
    FLT_track[Uidx]['X'].append(x_new)
    FLT_track[Uidx]['Y'].append(y_new)
    FLT_track[Uidx]['Z'].append(z_new)
    FLT_track[Uidx]['bearing'].append(effective_bearing)
    FLT_track[Uidx]['flight_mode'].append('soaring')
    FLT_track[Uidx]['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'])
    
    # Mettre à jour les conditions de vol avec les valeurs dynamiques
    FLT_conditions[Uidx]['airspeed'] = airspeed
    FLT_conditions[Uidx]['flight_path_angle'] = dynamic_flight_path_angle  # En radians
    FLT_conditions[Uidx]['bank_angle'] = target_bank_angle_rad  # En radians

    # Affichage en degrés pour le debug
    print(f"UAV {Uidx} soaring: new pos=({x_new:.1f}, {y_new:.1f}, {z_new:.1f}), "
          f"speed={airspeed:.1f}m/s, bank={np.degrees(target_bank_angle_rad):.1f}°, "
          f"γ={np.degrees(dynamic_flight_path_angle):.1f}°, bearing={np.degrees(effective_bearing):.1f}°")
    
    return FLT_track, FLT_conditions, wp_idx, SOAR_WPs

def EvalThermal(FLT_track, FLT_conditions, EVAL_WPs, nUAVs, Uidx, params, UAV_data, wp_idx, thermal_map, thermals):
    """
    Fonction pour évaluer les thermiques en suivant les points de passage EVAL_WPs.
    Le drone calcule la montée en altitude selon la puissance de la thermique.
    Tous les angles sont gérés en radians avec affichage en degrés.
    """
    LBz = params['Z_lower_bound']
    UBz = params['Z_upper_bound']
    t_step = params['time_step']
    
    # Obtenir les données de vol actuelles
    FLT_data = get_current_flight_data(FLT_track, FLT_conditions, nUAVs)

    if not FLT_track[Uidx]['in_evaluation']:
        return FLT_track, FLT_conditions, wp_idx, EVAL_WPs

    # Current position
    current_pos = {
        'X': FLT_data[Uidx]['X'],
        'Y': FLT_data[Uidx]['Y'],
        'Z': FLT_data[Uidx]['Z']
    }
    
    # Vérifier s'il y a des waypoints d'évaluation disponibles
    if not EVAL_WPs or len(EVAL_WPs['X']) == 0:
        print(f"UAV {Uidx}: Pas de waypoints d'évaluation disponibles")
        return FLT_track, FLT_conditions, wp_idx, EVAL_WPs
    
    # Vérifier si nous avons atteint la fin de la trajectoire
    if wp_idx >= len(EVAL_WPs['X']):
        print(f"UAV {Uidx}: Évaluation terminée - fin de trajectoire atteinte")
        return FLT_track, FLT_conditions, wp_idx, EVAL_WPs
    
    # Obtenir le waypoint cible actuel
    target_eval_wp = {
        'X': EVAL_WPs['X'][wp_idx],
        'Y': EVAL_WPs['Y'][wp_idx],
        'Z': EVAL_WPs['Z'][wp_idx]
    }
    
    # Calculer la distance au waypoint cible
    distance_to_eval_wp = np.sqrt((current_pos['X'] - target_eval_wp['X'])**2 + 
                                  (current_pos['Y'] - target_eval_wp['Y'])**2)
    
    # Distance parcourue par pas de temps (vitesse horizontale)
    eval_speed = FLT_data[Uidx]['airspeed']
    step_distance = eval_speed * params['time_step']

    # Vérifier si nous avons atteint le waypoint (seuil de proximité)
    proximity_threshold = max(10.0, step_distance * 0.8)

    print(f"UAV {Uidx}: Eval - wp_idx={wp_idx}/{len(EVAL_WPs['X'])}, "
          f"dist={distance_to_eval_wp:.1f}m, threshold={proximity_threshold:.1f}m")
    
    if distance_to_eval_wp <= proximity_threshold:
        # Waypoint atteint, passer au suivant
        wp_idx += 1
        print(f"UAV {Uidx}: Waypoint d'évaluation {wp_idx-1} atteint, passage au waypoint {wp_idx}")
        # Si nous avons atteint la fin de la trajectoire, retourner
        if wp_idx >= len(EVAL_WPs['X']):
            print(f"UAV {Uidx}: Évaluation terminée - tous les waypoints parcourus")
            return FLT_track, FLT_conditions, wp_idx, EVAL_WPs
    
    # Calculer le bearing vers la cible (en radians)
    bearing_to_target_rad = compute_bearing_cartesian(current_pos, target_eval_wp)[0]
    
    # Calculer la nouvelle position horizontale
    REF = dict()
    REF['X'] = FLT_data[Uidx]['X']
    REF['Y'] = FLT_data[Uidx]['Y']
    
    x_new, y_new = get_destination_from_range_and_bearing_cartesian(REF, step_distance, bearing_to_target_rad)
    
    z_new = FLT_data[Uidx]['Z']  # Altitude par défaut
    
    # Obtenir la thermique active pour calculer l'effet sur l'altitude
    thermal_id = FLT_track[Uidx]['current_thermal_id']
    if thermal_id is not None and thermal_id in thermals:
        thermal = thermals[thermal_id]
        
        # Vérifier si la thermique est encore active
        if thermal.is_active(params['current_simulation_time']):
            # Calculer la distance au centre de la thermique à la nouvelle position
            distance_from_center = np.sqrt((x_new - thermal.x)**2 + (y_new - thermal.y)**2)
            
            # Obtenir le taux de montée de la thermique à cette position
            thermal_lift_rate = thermal.get_lift_rate(distance_from_center)
            
            # Calculer le sink rate du drone selon le mode de vol
            if FLT_data[Uidx]['flight_mode'] == 'glide':
                # Mode glide : calculer le sink rate
                sink_rate = get_sink_rate(UAV_data, FLT_conditions[Uidx])
                
                # Calculer le taux de montée net
                net_climb_rate = thermal_lift_rate - sink_rate
                
                # Calculer le changement d'altitude
                altitude_change = net_climb_rate * t_step
                z_new = FLT_data[Uidx]['Z'] + altitude_change
                
                print(f"UAV {Uidx}: Mode glide - dist_center={distance_from_center:.1f}m, "
                      f"lift={thermal_lift_rate:.2f}m/s, sink={sink_rate:.2f}m/s, "
                      f"net_climb={net_climb_rate:.2f}m/s, Δh={altitude_change:.2f}m")
                
            elif FLT_data[Uidx]['flight_mode'] == 'engine':
                # Mode engine : le moteur peut compenser le sink rate, donc seul l'effet thermique compte
                altitude_change = thermal_lift_rate * t_step
                z_new = FLT_data[Uidx]['Z'] + altitude_change
                
                print(f"UAV {Uidx}: Mode engine - dist_center={distance_from_center:.1f}m, "
                      f"lift={thermal_lift_rate:.2f}m/s, Δh={altitude_change:.2f}m")
    
    # Limiter l'altitude aux bornes définies
    z_new = min(max(z_new, LBz), UBz)
    
    # Mettre à jour la trajectoire de vol (bearing en radians)
    FLT_track[Uidx]['X'].append(x_new)
    FLT_track[Uidx]['Y'].append(y_new)
    FLT_track[Uidx]['Z'].append(z_new)
    FLT_track[Uidx]['bearing'].append(bearing_to_target_rad)  # Stocker en radians
    
    # Mettre à jour les conditions de vol (angles en radians)
    FLT_conditions[Uidx]['airspeed'] = eval_speed
    FLT_conditions[Uidx]['flight_path_angle'] = 0.0

    # Mettre à jour la batterie selon le mode de vol
    if FLT_track[Uidx]['flight_mode'][-1] == 'engine':
        # Consommation en mode moteur
        pwr_level = get_power_consumption(UAV_data, FLT_conditions[Uidx])
        power_consumption_level = pwr_level * (t_step / 3600)
        new_battery = FLT_conditions[Uidx]['battery_capacity'] - power_consumption_level
    else:
        # Pas de consommation en mode glide
        new_battery = FLT_conditions[Uidx]['battery_capacity']

    # Déterminer le mode de vol pour le prochain pas
    if (FLT_track[Uidx]['flight_mode'][-1] == 'glide' and z_new <= LBz) or \
       (FLT_track[Uidx]['flight_mode'][-1] == 'engine' and z_new <= params['working_floor']):
        FLT_track[Uidx]['flight_mode'].append('engine')
    else:
        FLT_track[Uidx]['flight_mode'].append('glide')

    FLT_track[Uidx]['battery_capacity'].append(new_battery)
    FLT_conditions[Uidx]['battery_capacity'] = new_battery

    # Affichage en degrés pour le debug
    print(f"UAV {Uidx}: Eval pos=({x_new:.1f}, {y_new:.1f}, {z_new:.1f}), "
          f"bearing={np.degrees(bearing_to_target_rad):.1f}°, "
          f"γ={np.degrees(FLT_conditions[Uidx]['flight_path_angle']):.1f}°, "
          f"φ={np.degrees(FLT_conditions[Uidx]['bank_angle']):.1f}°, "
          f"mode={FLT_track[Uidx]['flight_mode'][-1]}")

    return FLT_track, FLT_conditions, wp_idx, EVAL_WPs


def gotoWaypointMulti(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data, current_wp_indices, current_eval_wp_indices, current_soar_wp_indices, thermal_map=None, thermal_evaluator=None, EVAL_WPs=None, thermals=None, SOAR_WPs=None):
    """
    Gère et contrôle plusieurs UAVs simultanément, en suivant et en mettant à jour leur progression vers leurs waypoints respectifs.

    Args:
        FLT_track (dict): Historique des positions et états des UAVs.
        FLT_conditions (dict): Conditions de vol actuelles des UAVs.
        GOAL_WPs (dict): Dictionnaire {Uidx: [ {'X', 'Y', 'Z'}] } pour chaque UAV.
        nUAVs (int): Nombre total d'UAVs.
        params (dict): Paramètres de simulation.
        UAV_data (dict): Paramètres physiques du UAV (identiques pour tous ou par UAV).
        current_wp_indices (dict): Dictionnaire {Uidx: index_wp_courant} pour chaque UAV.
        current_eval_wp_indices (dict): Dictionnaire {Uidx: index_wp_courant_evaluation} pour chaque UAV.
        current_soar_wp_indices (dict): Dictionnaire {Uidx: index_wp_courant_soaring} pour chaque UAV.
        thermal_map (ThermalMap, optional): Carte des thermiques. Defaults to None.
        thermal_evaluator (ThermalEvaluator, optional): Évaluateur de thermiques. Defaults to None.
        EVAL_WPs (dict, optional): Waypoints d'évaluation pour les thermiques. Defaults to None.
        active_thermals (dict, optional): Thermiques actifs. Defaults to None.
        SOAR_WPs (dict, optional): Waypoints de soaring. Defaults to None.

    Returns:
        tuple: (FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices, SOAR_WPs, current_soar_wp_indices) mis à jour pour tous les UAVs traités.
    """
    # Nettoyer les obstacles temporaires des UAVs qui ne sont plus en évaluation
    # Filtrer selon le type d'obstacle (dict vs array)
    cleaned_obstacles = []
    for obs in params['obstacles']:
        if isinstance(obs, dict):
            # Obstacle original (format dictionnaire) - garder si pas lié à un UAV ou si UAV encore en évaluation
            if obs.get('uav_id') is None or FLT_track[obs['uav_id']]['in_evaluation']:
                cleaned_obstacles.append(obs)
        else:
            # Obstacle converti (array numpy) - toujours garder (ce sont les obstacles fixes)
            cleaned_obstacles.append(obs)
    
    params['obstacles'] = cleaned_obstacles
    
    for Uidx in range(nUAVs):
        # Récupérer l'index du waypoint courant pour ce UAV
        print(f'in eval {FLT_track[Uidx]["in_evaluation"]}')
        if FLT_track[Uidx]['in_evaluation']:
            # UAV en mode évaluation
            wp_idx = current_eval_wp_indices[Uidx]
            
            # Vérifier si l'évaluation est terminée
            evaluation_complete = wp_idx >= len(EVAL_WPs[Uidx]['X'])
            
            if not evaluation_complete:
                # Continuer l'évaluation
                FLT_track, FLT_conditions, new_wp_idx, EVAL_WPs[Uidx] = EvalThermal(
                    FLT_track, FLT_conditions, EVAL_WPs[Uidx], nUAVs, Uidx, params, UAV_data, wp_idx, thermal_map, thermals
                )
                current_eval_wp_indices[Uidx] = new_wp_idx
                
                # Vérifier si cette étape termine l'évaluation
                if new_wp_idx >= len(EVAL_WPs[Uidx]['X']):
                    evaluation_complete = True
                    
            if evaluation_complete:
                FLT_track[Uidx]['in_evaluation'] = False
                if thermal_evaluator is not None and thermal_map is not None:
                    evaluation_result = thermal_evaluator.evaluate_thermal(thermal_map.detected_thermals[FLT_track[Uidx]['current_thermal_id']]['thermal'])
                    thermal_map.change_thermal_status(FLT_track[Uidx]['current_thermal_id'], evaluated=evaluation_complete, alt_gain=evaluation_result)
                    if evaluation_result:
                        FLT_track[Uidx]['flight_mode'].append('soaring')
                        FLT_track[Uidx]['soaring_start_time'] = params['current_simulation_time']
                        # Générer la trajectoire de soaring avec le bearing actuel
                        current_pos = {
                            'X': FLT_track[Uidx]['X'][-1],
                            'Y': FLT_track[Uidx]['Y'][-1],
                            'Z': FLT_track[Uidx]['Z'][-1]
                        }
                        active_thermals = thermal_map.get_active_thermals(params['current_simulation_time'])
                        thermal = active_thermals[FLT_track[Uidx]['current_thermal_id']]['thermal']
                        current_bearing = FLT_track[Uidx]['bearing'][-1]
                        
                        trajectory = thermal_evaluator.generate_soaring_trajectory(
                            current_pos, thermal, params['time_step'], 
                            FLT_conditions[Uidx], current_bearing
                        )
                        SOAR_WPs[Uidx] = trajectory
                        current_soar_wp_indices[Uidx] = 1
                    else:   
                        FLT_track[Uidx]['flight_mode'].append('glide')
                        FLT_track[Uidx]['current_thermal_id'] = None
        elif FLT_track[Uidx]['flight_mode'][-1] == 'soaring':
            wp_idx = current_soar_wp_indices[Uidx]
            FLT_track, FLT_conditions, new_wp_idx, SOAR_WPs[Uidx] = soarThermal(
                FLT_track, FLT_conditions, SOAR_WPs[Uidx], nUAVs, Uidx, params, UAV_data, wp_idx, thermal_map
            )
            current_soar_wp_indices[Uidx] = new_wp_idx
            # Vérifier les conditions de sortie
            thermal_id = FLT_track[Uidx]['current_thermal_id']
            should_exit = False
            active_thermals = thermal_map.get_active_thermals(params['current_simulation_time'])
            # Vérifier les conditions de sortie
            current_pos = {
                'X': FLT_track[Uidx]['X'][-1],
                'Y': FLT_track[Uidx]['Y'][-1],
                'Z': FLT_track[Uidx]['Z'][-1]
            }
            if thermal_id in active_thermals:
                thermal = active_thermals[thermal_id]['thermal']
                
                # Obtenir les positions des autres UAVs dans le même thermique
                other_uavs_in_thermal = []
                for other_idx in range(nUAVs):
                    if (other_idx != Uidx and 
                        FLT_track[other_idx]['flight_mode'][-1] == 'soaring' and
                        FLT_track[other_idx].get('current_thermal_id', None) == thermal_id):
                        other_uavs_in_thermal.append({
                            'X': FLT_track[other_idx]['X'][-1],
                            'Y': FLT_track[other_idx]['Y'][-1],
                            'Z': FLT_track[other_idx]['Z'][-1]
                        })
                
                
                soaring_start_time = FLT_track[Uidx].get('soaring_start_time', params['current_simulation_time'])
                
                should_exit = thermal_evaluator.check_soaring_exit_conditions(
                    current_pos, thermal, soaring_start_time, params['current_simulation_time'], 
                    other_uavs_in_thermal, SOAR_WPs[Uidx], current_soar_wp_indices[Uidx]
                )
            else:
                print(f"UAV {Uidx}: Thermique {thermal_id} non trouvée dans active_thermals, sortie forcée du soaring")
                should_exit = True
                exit_thermal = None
            if should_exit:
                FLT_track[Uidx]['flight_mode'].append('glide')
                FLT_track[Uidx]['soaring_start_time'] = None
                if thermal_id in active_thermals:
                    exit_thermal = thermal_map.detected_thermals[thermal_id]['thermal']
                    current_wp_indices[Uidx] = find_nearest_waypoint(
                        current_pos, GOAL_WPs[Uidx], params['obstacles'], exit_thermal, current_wp_indices[Uidx]
                    )
                else:
                   current_wp_indices[Uidx] = find_nearest_waypoint(
                       current_pos, GOAL_WPs[Uidx], params['obstacles'], None, current_wp_indices[Uidx]
                   )
        else:
            wp_idx = current_wp_indices[Uidx]
            FLT_track, FLT_conditions, new_wp_idx = gotoWaypoint(
                FLT_track, FLT_conditions, GOAL_WPs[Uidx], nUAVs, Uidx, params, UAV_data, wp_idx
            )
            current_wp_indices[Uidx] = new_wp_idx
            current_pos = {
                'X': FLT_track[Uidx]['X'][-1],
                'Y': FLT_track[Uidx]['Y'][-1],
                'Z': FLT_track[Uidx]['Z'][-1]
            }
            print("speed:", FLT_conditions[Uidx]['airspeed'])
            detected_thermal_id = detect_thermal_at_position(current_pos, thermals, params['current_simulation_time'])
            if detected_thermal_id is not None:
                if detected_thermal_id not in thermal_map.detected_thermals:
                    thermal_map.add_thermal_detection(detected_thermal_id, thermals[detected_thermal_id], params['current_simulation_time'])
                    #thermal_map.change_thermal_status(detected_thermal_id, True, True)
                    FLT_track[Uidx]['evaluation_start_altitude'] = current_pos['Z']

                    # Générér les Wps d'évaluation pour la thermique détectée
                    trajectoires = thermal_map.generate_evaluation_waypoints(current_pos, detected_thermal_id, FLT_conditions[Uidx]['airspeed'], FLT_track[Uidx]['bearing'][-1])
                    FLT_track[Uidx]['in_evaluation'] = True
                    FLT_track[Uidx]['current_thermal_id'] = detected_thermal_id
                    EVAL_WPs[Uidx]['X'] = trajectoires['X']
                    EVAL_WPs[Uidx]['Y'] = trajectoires['Y']
                    EVAL_WPs[Uidx]['Z'] = trajectoires['Z']
                    # Add circle to obstacles
                    active_thermals = thermal_map.get_active_thermals(params['current_simulation_time'])
                    thermal = active_thermals[FLT_track[Uidx]['current_thermal_id']]['thermal']
                    evaluation_obstacle = {
                        'X': thermal.x,
                        'Y': thermal.y,
                        'radius': thermal.radius,
                        'type': 'evaluation_zone',
                        'uav_id': Uidx
                    }
                    eval_poly = convert_cylindrical_obstacles_to_polygons([evaluation_obstacle])
                    # Add the evaluation obstacle to the list of obstacles
                    if 'obstacles' not in params:
                        params['obstacles'] = []
                    # Ajouter le polygone avec les métadonnées
                    obstacle_with_metadata = {
                        'vertices': eval_poly[0],
                        'uav_id': Uidx,
                        'thermal_id': detected_thermal_id,
                        'type': 'evaluation_zone'
                    }
                    params['obstacles'].append(obstacle_with_metadata)
                else:
                    thermal = thermal_map.detected_thermals[detected_thermal_id]['thermal']
                    if thermal.is_active(params['current_simulation_time']) and \
                       FLT_track[Uidx]['current_thermal_id'] != detected_thermal_id and \
                       thermal_map.detected_thermals[detected_thermal_id]['evaluated'] and \
                       thermal_map.detected_thermals[detected_thermal_id]['alt_gain']:
                        
                        # Utiliser directement pour le soaring
                        FLT_track[Uidx]['current_thermal_id'] = detected_thermal_id
                        FLT_track[Uidx]['flight_mode'].append('soaring')
                        if len(FLT_track[Uidx]['flight_mode']) == 1 or FLT_track[Uidx]['flight_mode'][-2] != 'soaring':
                            FLT_track[Uidx]['soaring_start_time'] = params['current_simulation_time']
                        current_soar_wp_indices[Uidx] = 1
                        trajectory = thermal_evaluator.generate_soaring_trajectory(current_pos, thermal, params['time_step'], FLT_conditions[Uidx], FLT_track[Uidx]['bearing'][-1])
                        SOAR_WPs[Uidx] = trajectory

    return FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices, SOAR_WPs, current_soar_wp_indices