import numpy as np
import warnings
from compute import (
    calculate_optimal_climb_angle,
    check_segment_obstacle_collision,
    compute_distance_cartesian,
    decision_making,
    find_nearest_waypoint,
    get_current_flight_data,
    get_power_consumption,
    get_sink_rate,
    get_destination_from_range_and_bearing_cartesian,
    is_point_in_obstacle
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


def gotoWaypoint(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, Uidx, params, UAV_data, current_wp_idx, thermal_map=None, thermal_evaluator=None):
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
    if distance_to_wp <= 10 or next_step_distance >= distance_to_wp:
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
                
    elif FLT_data[Uidx]['flight_mode'] == 'soaring':
        if thermal_map is not None and thermal_evaluator is not None:
            # Obtenir le thermique exploité
            thermal_id = FLT_track[Uidx].get('current_thermal_id', None)
            if thermal_id is not None:
                active_thermals = thermal_map.get_active_thermals(Tsim_current)
                
                if thermal_id in active_thermals:
                    thermal = active_thermals[thermal_id]['thermal']
                    
                    # Générer des candidats de soaring (différentes positions dans la spirale)
                    soaring_candidates = []

                    # Option 1: Position actuelle dans la spirale
                    soaring_result_1 = thermal_evaluator.generate_soaring_trajectory(
                        current_pos, thermal, Tsim_current, t_step
                    )
                    soaring_candidates.append(soaring_result_1)

                    # Option 2: Ajuster légèrement la position dans la spirale (avancer un peu)
                    adjusted_pos_1 = current_pos.copy()
                    angle_offset = 0.1  # radians
                    current_angle = np.arctan2(current_pos['Y'] - thermal.y, current_pos['X'] - thermal.x)
                    spiral_radius = thermal.radius * 0.6
                    adjusted_pos_1['X'] = thermal.x + spiral_radius * np.cos(current_angle + angle_offset)
                    adjusted_pos_1['Y'] = thermal.y + spiral_radius * np.sin(current_angle + angle_offset)

                    soaring_result_2 = thermal_evaluator.generate_soaring_trajectory(
                        adjusted_pos_1, thermal, Tsim_current, t_step
                    )
                    soaring_candidates.append(soaring_result_2)

                    # Option 3: Ajuster dans l'autre direction
                    adjusted_pos_2 = current_pos.copy()
                    adjusted_pos_2['X'] = thermal.x + spiral_radius * np.cos(current_angle - angle_offset)
                    adjusted_pos_2['Y'] = thermal.y + spiral_radius * np.sin(current_angle - angle_offset)

                    soaring_result_3 = thermal_evaluator.generate_soaring_trajectory(
                        adjusted_pos_2, thermal, Tsim_current, t_step
                    )
                    soaring_candidates.append(soaring_result_3)

                    # Ajouter tous les candidats de soaring
                    for soaring_result in soaring_candidates:
                        candidate_sol['X'].append(soaring_result['X'])
                        candidate_sol['Y'].append(soaring_result['Y'])
                        candidate_sol['Z'].append(min(soaring_result['Z'], UBz))
                        candidate_sol['bearing'].append(FLT_data[Uidx]['bearing'])  # Maintenir le cap actuel
                        candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'])  # Pas de consommation
                        candidate_sol['flight_mode'].append('soaring')
                        candidate_sol['airspeed'].append(min_velocity)  # Vitesse minimale en soaring
                        candidate_sol['flight_path_angle'].append(np.arctan(soaring_result['climb_rate'] / min_velocity))

                        # Pour la prédiction, rester dans le thermique
                        PB_temp['X'].append(soaring_result['X'])
                        PB_temp['Y'].append(soaring_result['Y'])
                        PB_temp['Z'].append(min(soaring_result['Z'] + 50, UBz))  # Prédire une montée continue


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
        elif FLT_data[u]['flight_mode'] == 'soaring':
            pridction_distance = 50
        
        REF = dict()
        REF['X'] = FLT_data[u]['X']
        REF['Y'] = FLT_data[u]['Y']
        x_obs, y_obs = get_destination_from_range_and_bearing_cartesian(REF, pridction_distance, FLT_data[u]['bearing'])
        Dist2Horizon['X'].append(x_obs)
        Dist2Horizon['Y'].append(y_obs)
        if FLT_data[u]['flight_mode'] == 'soaring':
            Dist2Horizon['Z'].append(min(FLT_data[u]['Z'] + 50, UBz))
        else:
            Dist2Horizon['Z'].append(LBz)

    x0, y0, z0 = FLT_data[Uidx]['X'], FLT_data[Uidx]['Y'], FLT_data[Uidx]['Z']

    for i in range(len(H)*len(V)):
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
    
    for i in range(len(H)*len(V)):
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
    
    selected_mode = candidate_sol['flight_mode'][idx]
    
    if selected_mode == 'soaring' and thermal_map is not None and thermal_evaluator is not None:
        thermal_id = FLT_track[Uidx].get('current_thermal_id', None)
        if thermal_id is not None:
            active_thermals = thermal_map.get_active_thermals(Tsim_current)
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
                
                # Vérifier les conditions de sortie
                current_selected_pos = {
                    'X': candidate_sol['X'][idx],
                    'Y': candidate_sol['Y'][idx],
                    'Z': candidate_sol['Z'][idx]
                }
                soaring_start_time = FLT_track[Uidx].get('soaring_start_time', Tsim_current)
                should_exit = thermal_evaluator.check_soaring_exit_conditions(
                    current_selected_pos, thermal, soaring_start_time, Tsim_current, other_uavs_in_thermal
                )
                
                if should_exit:
                    selected_mode = 'glide'
                    FLT_track[Uidx]['current_thermal_id'] = None
                    FLT_track[Uidx]['soaring_start_time'] = None
                    current_wp_idx = find_nearest_waypoint(current_pos, GOAL_WPs)
            else:
                # Le thermique n'est plus actif
                selected_mode = 'glide'
                FLT_track[Uidx]['current_thermal_id'] = None
                FLT_track[Uidx]['soaring_start_time'] = None
                current_wp_idx = find_nearest_waypoint(current_pos, GOAL_WPs)
        else:
            # Pas de thermique associé
            selected_mode = 'glide'
            current_wp_idx = find_nearest_waypoint(current_pos, GOAL_WPs)

    if selected_mode != 'soaring':
        # Si le mode n'est pas soaring, on vérifie les conditions de vol
        # Si engine et altitude plus petit que working floor ou si glide et altitude plus petit que LBz alors engine else glide
        if (selected_mode == 'engine' and FLT_track[Uidx]['Z'][-1] <= params['working_floor']) or \
           (selected_mode == 'glide' and FLT_track[Uidx]['Z'][-1] <= LBz):
            selected_mode = 'engine'
        else:
            selected_mode = 'glide'

    FLT_track[Uidx]['battery_capacity'].append(candidate_sol['battery_capacity'][idx])

    FLT_conditions[Uidx]['airspeed'] = candidate_sol['airspeed'][idx]
    FLT_conditions[Uidx]['flight_path_angle'] = candidate_sol['flight_path_angle'][idx]

    return FLT_track, FLT_conditions, current_wp_idx

def gotoWaypointMulti(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data, current_wp_indices, current_eval_wp_indices, thermal_map=None, thermal_evaluator=None, EVAL_WPs=None):
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
        print(f'in eval {FLT_track[Uidx]["in_evaluation"]}')
        if FLT_track[Uidx]['in_evaluation']:
            # UAV en mode évaluation
            wp_idx = current_eval_wp_indices[Uidx]
            
            # Vérifier si l'évaluation est terminée
            evaluation_complete = wp_idx >= len(EVAL_WPs[Uidx]['X'])
            
            if not evaluation_complete:
                # Continuer l'évaluation
                FLT_track, FLT_conditions, new_wp_idx = gotoWaypoint(
                    FLT_track, FLT_conditions, EVAL_WPs[Uidx], nUAVs, Uidx, params, UAV_data, wp_idx, thermal_map
                )
                current_eval_wp_indices[Uidx] = new_wp_idx
                
                # Vérifier si cette étape termine l'évaluation
                if new_wp_idx >= len(EVAL_WPs[Uidx]['X']):
                    evaluation_complete = True
                    
            if evaluation_complete:
                FLT_track[Uidx]['in_evaluation'] = False
                if thermal_evaluator is not None and thermal_map is not None:
                    evaluation_result = thermal_evaluator.evaluate_thermal(FLT_track[Uidx], len(EVAL_WPs[Uidx]['X']))
                    thermal_map.change_thermal_status(FLT_track[Uidx]['current_thermal_id'], evaluated=evaluation_complete, alt_gain=evaluation_result)
                    if evaluation_result:
                        FLT_track[Uidx]['flight_mode'][-1] = 'soaring'
                        FLT_track[Uidx]['soaring_start_time'] = params['current_simulation_time']
                    else:   
                        FLT_track[Uidx]['flight_mode'][-1] = 'glide'
                        FLT_track[Uidx]['current_thermal_id'] = None
                    
        else:
            wp_idx = current_wp_indices[Uidx]
            FLT_track, FLT_conditions, new_wp_idx = gotoWaypoint(
                FLT_track, FLT_conditions, GOAL_WPs[Uidx], nUAVs, Uidx, params, UAV_data, wp_idx, thermal_map, thermal_evaluator
            )
            current_wp_indices[Uidx] = new_wp_idx
            
            
    return FLT_track, FLT_conditions, current_wp_indices, current_eval_wp_indices