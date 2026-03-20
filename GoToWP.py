import numpy as np
import warnings
from compute import (
    calculate_optimal_climb_angle,
    check_segment_obstacle_collision,
    compute_distance_cartesian,
    decision_making,
    get_current_flight_data,
    get_power_consumption,
    get_sink_rate,
    get_destination_from_range_and_bearing_cartesian,
    is_point_in_obstacle
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import copy

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
    Handles parallel lines and near-parallel cases gracefully.

    Args:
        pA (np.ndarray): Points defining the first line (2x3).
        pB (np.ndarray): Points defining the second line (2x3).

    Returns:
        np.ndarray: Coordinates of the intersection point (3x1), or np.inf if lines are parallel.
    """
    try:
        SI = pA - pB
        
        # Vérifier si les points sont identiques ou très proches
        if np.allclose(SI, 0, atol=1e-6):
            return np.array([[np.inf], [np.inf], [np.inf]])
        
        norms = np.sqrt(np.sum(np.power(SI, 2), axis=0))
        
        # Vérifier si les normes sont valides
        if np.any(norms < 1e-6):
            return np.array([[np.inf], [np.inf], [np.inf]])
        
        NI = np.divide(SI, norms)
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
        
        # Vérifier si la matrice est singulière (déterminant proche de 0)
        det = np.linalg.det(S)
        if abs(det) < 1e-10:
            # Lignes parallèles ou presque parallèles
            return np.array([[np.inf], [np.inf], [np.inf]])
        
        CX = np.sum(np.multiply(pA[:, 0], np.power(nX, 2) - 1) + 
                   np.multiply(pA[:, 1], np.multiply(nX, nY)) + 
                   np.multiply(pA[:, 2], np.multiply(nX, nZ)))
        CY = np.sum(np.multiply(pA[:, 0], np.multiply(nX, nY)) + 
                   np.multiply(pA[:, 1], np.power(nY, 2) - 1) + 
                   np.multiply(pA[:, 2], np.multiply(nY, nZ)))
        CZ = np.sum(np.multiply(pA[:, 0], np.multiply(nX, nZ)) + 
                   np.multiply(pA[:, 1], np.multiply(nY, nZ)) + 
                   np.multiply(pA[:, 2], np.power(nZ, 2) - 1))
        
        C = np.array([[CX], [CY], [CZ]])
        
        # Résoudre le système linéaire
        P_intersect = np.linalg.solve(S, C)
        
        # Vérifier si le résultat est valide
        if np.any(np.isnan(P_intersect)) or np.any(np.isinf(P_intersect)):
            return np.array([[np.inf], [np.inf], [np.inf]])
        
        return P_intersect
        
    except np.linalg.LinAlgError:
        # Matrice singulière : lignes parallèles
        return np.array([[np.inf], [np.inf], [np.inf]])
    except Exception as e:
        # Toute autre erreur
        return np.array([[np.inf], [np.inf], [np.inf]])


def gotoWaypoint(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, Uidx, params, UAV_data, current_wp_idx):
    """
    Guides a UAV towards its next waypoint while considering flight dynamics, energy constraints, and collision avoidance.
    """
    
    # Vérifier si le waypoint actuel est valide
    if current_wp_idx >= len(GOAL_WPs['X']):
        print(f"UAV {Uidx}: Tous les waypoints atteints ({current_wp_idx}/{len(GOAL_WPs['X'])})")
        # Retourner sans modification - l'UAV a terminé sa mission
        return FLT_track, FLT_conditions, current_wp_idx

    if 'collisions_avoided_count' not in FLT_track[Uidx]:
        FLT_track[Uidx]['collisions_avoided_count'] = 0
    
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
    # Coefficient alpha pour augmenter la priorité d'un critère (>1)
    alpha = params['alpha']
    # Sélectionner les steps en fonction du mode de vol actuel
    if FLT_track[Uidx]['flight_mode'][-1] == 'glide':
        h_step = params['bearing_step_glide']
        v_step = params['speed_step_glide']
    else:  # 'engine' ou autre mode
        h_step = params['bearing_step_engine']
        v_step = params['speed_step_engine']
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

    #Créer REF une seule fois au lieu de le recréer dans chaque itération
    REF = {
        'X': FLT_data[Uidx]['X'],
        'Y': FLT_data[Uidx]['Y']
    }

    if FLT_data[Uidx]['flight_mode'] == 'glide':
        # Pré-calculer les valeurs communes
        current_z = FLT_data[Uidx]['Z']
        delta_z = abs(current_z - LBz)
        
        for i in range(len(H)):
            for j in range(len(V)):
                FLT_conditions[Uidx]['airspeed'] = V[j]
                FLT_conditions[Uidx]['airspeed_dot'] = 0.0
                dZ = get_sink_rate(UAV_data, FLT_conditions[Uidx])
                pridction_distance = (delta_z/dZ)*V[j] if dZ != 0 else V[j] * t_step
                TD = V[j] * t_step
                dZ_sink = -dZ * t_step

                x_new, y_new = get_destination_from_range_and_bearing_cartesian(REF, TD, H[i])

                alt = min(max(current_z + dZ_sink, LBz), UBz)

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
        
        # VALIDATION: Vérifier que climb_angle est valide
        if np.isnan(climb_angle) or np.isinf(climb_angle):
            print(f"⚠️  UAV {Uidx}: climb_angle invalide ({climb_angle}), utilisation angle par défaut de 5°")
            climb_angle = np.deg2rad(5.0)  # Angle de montée par défaut
        
        level_angle = 0
        #descent_angle = np.deg2rad(-5)
        
        # Pré-calculer les constantes trigonométriques et valeurs communes
        cos_climb = np.cos(climb_angle)
        sin_climb = np.sin(climb_angle)
        #cos_descent = np.cos(descent_angle)
        #sin_descent = np.sin(descent_angle)
        current_z = FLT_data[Uidx]['Z']
        delta_z_up = abs(UBz - current_z)
        delta_z_down = abs(current_z - LBz)

        # Pour chaque cap possible
        for i in range(len(H)):
            for j in range(len(V)):
                # VALIDATION: Vérifier que V[j] est valide
                if np.isnan(V[j]) or np.isinf(V[j]) or V[j] <= 0:
                    print(f"⚠️  UAV {Uidx}: V[{j}]={V[j]} invalide, skip")
                    continue
                
                if np.isnan(H[i]) or np.isinf(H[i]):
                    print(f"⚠️  UAV {Uidx}: H[{i}]={H[i]} invalide, skip")
                    continue
                
                FLT_conditions[Uidx]['airspeed'] = V[j]
                FLT_conditions[Uidx]['airspeed_dot'] = 0.0
                
                # 1. MONTÉE AVEC MOTEUR
                FLT_conditions[Uidx]['flight_path_angle'] = climb_angle
                
                # Calculer la distance parcourue et le changement d'altitude (utiliser constantes pré-calculées)
                horizontal_distance = V[j] * t_step * cos_climb
                
                # VALIDATION: Vérifier que horizontal_distance est valide
                if np.isnan(horizontal_distance) or np.isinf(horizontal_distance):
                    print(f"⚠️  UAV {Uidx}: horizontal_distance={horizontal_distance} invalide (V={V[j]}, t_step={t_step}, cos_climb={cos_climb}), skip")
                    continue
                altitude_change = V[j] * t_step * sin_climb
                
                # Calculer la consommation d'énergie pour la montée
                pwr_climb = get_power_consumption(UAV_data, FLT_conditions[Uidx])
                power_consumption_climb = pwr_climb * (t_step / 3600)
                
                # Calculer la nouvelle position (réutiliser REF)
                x_new_climb, y_new_climb = get_destination_from_range_and_bearing_cartesian(REF, horizontal_distance, H[i])
                
                # Validation
                if np.isnan(x_new_climb) or np.isnan(y_new_climb):
                    print(f"UAV {Uidx} ENGINE-CLIMB: NaN position - dist={horizontal_distance:.2f}, H[{i}]={H[i]:.3f}")
                    continue
                
                alt_climb = min(max(FLT_data[Uidx]['Z'] + altitude_change, LBz), UBz)
                
                if np.isnan(alt_climb):
                    print(f"UAV {Uidx} ENGINE-CLIMB: NaN altitude - Z={FLT_data[Uidx]['Z']:.1f}, change={altitude_change:.2f}")
                    continue
                
                # Ajouter le candidat pour la montée
                candidate_sol['X'].append(x_new_climb)
                candidate_sol['Y'].append(y_new_climb)
                candidate_sol['Z'].append(alt_climb)
                candidate_sol['bearing'].append(H[i])
                candidate_sol['battery_capacity'].append(FLT_data[Uidx]['battery_capacity'] - power_consumption_climb)
                candidate_sol['flight_mode'].append('engine')
                candidate_sol['airspeed'].append(V[j])
                candidate_sol['flight_path_angle'].append(climb_angle)
                
                # Prédiction pour collision (réutiliser delta_z_up)
                pridction_distance = (delta_z_up / altitude_change) * horizontal_distance if abs(altitude_change) > 1e-6 else V[j] * t_step
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
                
                # Validation
                if np.isnan(x_new_level) or np.isnan(y_new_level):
                    print(f"UAV {Uidx} ENGINE-LEVEL: NaN position - dist={horizontal_distance:.2f}, H[{i}]={H[i]:.3f}")
                    continue
                
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
        
        else:
            # Pour les modes 'soar', 'eval', ou autres : prédiction simple basée sur la vitesse
            pridction_distance = FLT_data[u]['airspeed'] * t_step
        
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
            if is_point_in_obstacle(candidate_point, obstacle):
                flag3 = False
                break
                
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

            # Vérifier si les trajectoires sont trop similaires pour calculer l'intersection
            traj_uav1 = np.array([xx0 - x0, yy0 - y0, zz0 - z0])
            traj_uav2 = np.array([xx1 - x1, yy1 - y1, zz1 - z1])
            
            # Normaliser les vecteurs de trajectoire
            norm1 = np.linalg.norm(traj_uav1)
            norm2 = np.linalg.norm(traj_uav2)
            
            # Si l'une des trajectoires est nulle ou les trajectoires sont parallèles
            if norm1 < 1e-6 or norm2 < 1e-6:
                # Utiliser la distance directe
                if D < safe_dist:
                    flag2 = False
                continue
            
            traj_uav1_norm = traj_uav1 / norm1
            traj_uav2_norm = traj_uav2 / norm2
            
            # Calculer l'angle entre les trajectoires
            dot_product = np.dot(traj_uav1_norm, traj_uav2_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # Éviter les erreurs d'arrondi
            
            # Si les trajectoires sont parallèles ou presque (angle < 5 degrés)
            if abs(dot_product) > 0.996:  # cos(5°) ≈ 0.996
                # Calculer la distance perpendiculaire entre les lignes parallèles
                point_diff = np.array([x1 - x0, y1 - y0, z1 - z0])
                cross_product = np.cross(traj_uav1_norm, point_diff)
                perpendicular_dist = np.linalg.norm(cross_product)
                
                if perpendicular_dist < safe_dist:
                    flag2 = False
                continue

            pA = np.array([[x0, y0, z0], [x1, y1, z1]])
            pB = np.array([[xx0, yy0, zz0], [xx1, yy1, zz1]])
            P_intersect = lineXline(pA, pB)

            if not(all(np.isinf(P_intersect))):
                x_int, y_int, z_int = P_intersect[0, 0], P_intersect[1, 0], P_intersect[2, 0]

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
                DT = abs(t1 - t2) * candidate_sol['airspeed'][i]
                flag2 = flag2 and (D0 >= HorizonLength or (D0 < HorizonLength and DT >= safe_dist))

        VO_flag.append(flag1)
        PO_flag.append(flag2)
        OB_flag.append(flag3)

    # M8b: un événement d'évitement est compté si au moins un candidat a été
    # rejeté pour risque inter-agent mais que le candidat choisi est sûr.
    safety_rejected_candidates = sum(
        1 for i in range(len(VO_flag)) if not (VO_flag[i] and PO_flag[i])
    )

    # Vérifier qu'il y a des candidats valides
    if len(candidate_sol['X']) == 0:
        print(f"❌ UAV {Uidx}: Aucun candidat valide généré!")
        print(f"   Mode: {FLT_data[Uidx]['flight_mode']}, Position: ({FLT_data[Uidx]['X']:.1f}, {FLT_data[Uidx]['Y']:.1f}, {FLT_data[Uidx]['Z']:.1f})")
        # Conserver position actuelle
        return FLT_track, FLT_conditions, current_wp_idx
    
    # Vectorisation optimale des calculs de coûts
    # Pré-calculer les valeurs constantes
    dest = {
        'X': GOAL_WPs['X'][current_wp_idx],
        'Y': GOAL_WPs['Y'][current_wp_idx],
        'Z': GOAL_WPs['Z'][current_wp_idx]
    }
    battery_cap = FLT_data[Uidx]['battery_capacity']
    z_ref = FLT_data[Uidx]['Z']
    
    # Construire un seul array 2D pour les positions
    pos_array = np.column_stack([
        candidate_sol['X'],
        candidate_sol['Y'],
        candidate_sol['Z']
    ])
    dest_array = np.array([dest['X'], dest['Y'], dest['Z']])
    
    # Calcul vectorisé optimisé des distances avec np.linalg.norm
    C_distance = np.linalg.norm(pos_array - dest_array, axis=1)
    
    # Opérations booléennes vectorisées (pas de list comprehension)
    VO_flag_array = np.array(VO_flag, dtype=bool)
    PO_flag_array = np.array(PO_flag, dtype=bool)
    OB_flag_array = np.array(OB_flag, dtype=bool)
    
    C_safety = -((VO_flag_array & PO_flag_array).astype(float))
    C_obstacle = -(OB_flag_array.astype(float))
    
    # Calculs vectorisés pour énergie et sink
    battery_array = np.array(candidate_sol['battery_capacity'])
    C_energy = (battery_cap - battery_array) / battery_cap
    C_sink = (z_ref - pos_array[:, 2]) / z_ref
    
    # Matrice de décision avec pondération alpha pour prioriser certains critères
    # alpha > 1 : augmente l'importance du critère pondéré
    # Mode glide : alpha pondère C_sink (minimiser la descente)
    # Mode engine : alpha pondère C_energy (minimiser la consommation)
    if FLT_data[Uidx]['flight_mode'] == 'glide' :
        DM = np.column_stack([C_safety, C_distance, C_energy, alpha*C_sink, C_obstacle])
    elif FLT_data[Uidx]['flight_mode'] == 'engine' :
        DM = np.column_stack([C_safety, C_distance, alpha*C_energy, C_sink, C_obstacle])
    ranked_indices = decision_making(DM)
    idx = ranked_indices[0]  # Meilleur candidat selon le classement
    if safety_rejected_candidates > 0 and VO_flag[idx] and PO_flag[idx]:
        FLT_track[Uidx]['collisions_avoided_count'] += 1
    
    # VALIDATION FINALE : Vérifier que la solution choisie est valide
    final_x = candidate_sol['X'][idx]
    final_y = candidate_sol['Y'][idx]
    final_z = candidate_sol['Z'][idx]
    final_bearing = candidate_sol['bearing'][idx]
    
    if np.isnan(final_x) or np.isnan(final_y) or np.isnan(final_z) or np.isnan(final_bearing):
        print(f"❌ UAV {Uidx}: Solution finale invalide détectée!")
        print(f"   Pos: ({final_x:.2f}, {final_y:.2f}, {final_z:.2f}), Bearing: {final_bearing:.3f}")
        print(f"   Mode: {candidate_sol['flight_mode'][idx]}, Airspeed: {candidate_sol['airspeed'][idx]:.2f}")
        print(f"   Nombre de candidats: {len(candidate_sol['X'])}, Index choisi: {idx}")
        
        # En cas de solution invalide, conserver la dernière position valide
        if len(FLT_track[Uidx]['X']) > 0:
            print(f"   ➡️  Conservation dernière position valide")
            return FLT_track, FLT_conditions, current_wp_idx
        else:
            print(f"   ❌ Aucune position valide précédente - Erreur critique")
            return FLT_track, FLT_conditions, current_wp_idx

    FLT_track[Uidx]['X'].append(final_x)
    FLT_track[Uidx]['Y'].append(final_y)
    FLT_track[Uidx]['Z'].append(final_z)
    FLT_track[Uidx]['bearing'].append(final_bearing)
    
    candidate_mode = candidate_sol['flight_mode'][idx]

    # Si le mode n'est pas soaring, on vérifie les conditions de vol
    # Si engine et altitude plus petit que working floor ou si glide et altitude plus petit que LBz alors engine else glide
    if (candidate_mode == 'engine' and FLT_track[Uidx]['Z'][-1] <= params['working_floor'] and FLT_track[Uidx]['current_thermal_id'] is None) or \
       (candidate_mode == 'glide' and FLT_track[Uidx]['Z'][-1] <= LBz and FLT_track[Uidx]['current_thermal_id'] is None):
        FLT_track[Uidx]['flight_mode'].append('engine')
    else:
        FLT_track[Uidx]['flight_mode'].append('glide')

    FLT_track[Uidx]['battery_capacity'].append(candidate_sol['battery_capacity'][idx])
    
    # Mettre à jour le temps de vol
    if len(FLT_track[Uidx]['flight_time']) > 0:
        FLT_track[Uidx]['flight_time'].append(FLT_track[Uidx]['flight_time'][-1] + params['time_step'])
    else:
        FLT_track[Uidx]['flight_time'].append(params['time_step'])

    FLT_conditions[Uidx]['airspeed'] = candidate_sol['airspeed'][idx]
    FLT_conditions[Uidx]['flight_path_angle'] = candidate_sol['flight_path_angle'][idx]
    
    return FLT_track, FLT_conditions, current_wp_idx


def process_single_uav(Uidx, FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data, 
                       current_wp_indices):
    """
    Traite un seul UAV - fonction auxiliaire pour la parallélisation.
    L'évaluation et le soaring des thermiques sont gérés par MAVSDK (orbit/loiter)
    dans la boucle principale de dronePx4.py, pas ici.
    """
    
    result = {
        'Uidx': Uidx,
        'track': None,
        'conditions': None,
        'wp_idx': None,
        'mission_complete': False
    }
    
    try:
        # Vérifier si la mission est déjà terminée
        if current_wp_indices[Uidx] >= len(GOAL_WPs[Uidx]['X']):
            print(f"UAV {Uidx}: Mission déjà terminée")
            result['mission_complete'] = True
            result['wp_idx'] = current_wp_indices[Uidx]
            return result
        
        # Si le drone est en évaluation ou soaring, ne pas traiter ici
        # C'est géré par MAVSDK orbit/loiter dans dronePx4.py
        if FLT_track[Uidx]['in_evaluation']:
            result['wp_idx'] = current_wp_indices[Uidx]
            result['track'] = FLT_track[Uidx]
            result['conditions'] = FLT_conditions[Uidx]
            return result
        
        if FLT_track[Uidx]['flight_mode'][-1] == 'soaring':
            result['wp_idx'] = current_wp_indices[Uidx]
            result['track'] = FLT_track[Uidx]
            result['conditions'] = FLT_conditions[Uidx]
            return result
        
        # Si le drone est en atterrissage / atterri, ne pas traiter
        if FLT_track[Uidx]['flight_mode'][-1] == 'landing':
            result['wp_idx'] = current_wp_indices[Uidx]
            result['track'] = FLT_track[Uidx]
            result['conditions'] = FLT_conditions[Uidx]
            return result
        
        # Créer une structure temporaire - copier SEULEMENT l'UAV actuel (optimisation)
        temp_track = {}
        temp_conditions = {}
        
        # Copie profonde uniquement pour l'UAV traité
        temp_track[Uidx] = copy.deepcopy(FLT_track[Uidx])
        temp_conditions[Uidx] = copy.deepcopy(FLT_conditions[Uidx])
        
        # Références directes pour les autres (lecture seule)
        for i in range(nUAVs):
            if i != Uidx:
                temp_track[i] = FLT_track[i]
                temp_conditions[i] = FLT_conditions[i]
        
        wp_idx = current_wp_indices[Uidx]
        
        temp_track, temp_conditions, new_wp_idx = gotoWaypoint(
            temp_track, temp_conditions, GOAL_WPs[Uidx], nUAVs, Uidx, 
            params, UAV_data, wp_idx
        )
        result['wp_idx'] = new_wp_idx
        
        # Vérifier si l'UAV vient d'atteindre son dernier waypoint
        if new_wp_idx >= len(GOAL_WPs[Uidx]['X']):
            print(f"UAV {Uidx}: Mission terminée - dernier waypoint atteint")
            result['mission_complete'] = True
        
        # Extraire uniquement les données de l'UAV traité
        result['track'] = temp_track[Uidx]
        result['conditions'] = temp_conditions[Uidx]
        
        # Validation : Vérifier qu'il n'y a pas de NaN dans les résultats
        if len(temp_track[Uidx]['X']) > 0:
            last_x = temp_track[Uidx]['X'][-1]
            last_y = temp_track[Uidx]['Y'][-1]
            last_z = temp_track[Uidx]['Z'][-1]
            
            if np.isnan(last_x) or np.isnan(last_y) or np.isnan(last_z):
                print(f"⚠️  UAV {Uidx}: Position invalide générée - annulation")
                result['error'] = 'invalid_position'
                result['track'] = None
                result['conditions'] = None
        
    except Exception as e:
        import traceback
        print(f"Erreur lors du traitement de l'UAV {Uidx}: {e}")
        print(f"Traceback complet:\n{traceback.format_exc()}")
        result['error'] = str(e)
    
    return result


def gotoWaypointMulti(FLT_track, FLT_conditions, GOAL_WPs, nUAVs, params, UAV_data, 
                      current_wp_indices, max_workers=None):
    """
    Version parallélisée de gotoWaypointMulti utilisant ThreadPoolExecutor.
    Chaque drone est traité de manière totalement indépendante et peut avancer à son propre rythme.
    
    La gestion des thermiques (détection, évaluation, soaring) est désormais gérée 
    par MAVSDK orbit/loiter dans la boucle principale de dronePx4.py.
    
    Args:
        max_workers (int, optional): Nombre maximum de threads. Par défaut: nUAVs (un thread par drone)
    """
    # Nettoyer obstacles temporaires (supprimer ceux dont le UAV n'est plus en évaluation)
    cleaned_obstacles = []
    for obs in params['obstacles']:
        if isinstance(obs, dict):
            if obs.get('uav_id') is None or FLT_track[obs['uav_id']]['in_evaluation']:
                cleaned_obstacles.append(obs)
        else:
            cleaned_obstacles.append(obs)
    
    params['obstacles'] = cleaned_obstacles
    
    # Déterminer nombre de workers - un thread par drone pour indépendance maximale
    if max_workers is None:
        max_workers = nUAVs
    
    # Créer fonction partielle avec paramètres communs
    process_func = partial(
        process_single_uav,
        FLT_track=FLT_track,
        FLT_conditions=FLT_conditions,
        GOAL_WPs=GOAL_WPs,
        nUAVs=nUAVs,
        params=params,
        UAV_data=UAV_data,
        current_wp_indices=current_wp_indices
    )
    
    # Traiter tous les UAVs en parallèle - chaque drone avance à son rythme
    results = []
    failed_uavs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les drones en parallèle
        futures = {executor.submit(process_func, Uidx): Uidx for Uidx in range(nUAVs)}
        
        # Collecter les résultats au fur et à mesure
        for future in as_completed(futures):
            Uidx = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"⚠️ Exception UAV {Uidx}: {e}")
                failed_uavs.append(Uidx)
                results.append({'Uidx': Uidx, 'error': str(e), 'wp_idx': current_wp_indices[Uidx]})
    
    if failed_uavs:
        print(f"ℹ️ {len(failed_uavs)}/{nUAVs} drones ont rencontré des erreurs: {failed_uavs}")
    
    # Appliquer les résultats dans l'ordre
    results.sort(key=lambda x: x['Uidx'])
    
    for result in results:
        if 'error' in result:
            continue
            
        Uidx = result['Uidx']
        
        # Mettre à jour track et conditions
        if result['track'] is not None:
            FLT_track[Uidx] = result['track']
        if result['conditions'] is not None:
            FLT_conditions[Uidx] = result['conditions']
        
        # Mettre à jour indices waypoint
        if result['wp_idx'] is not None:
            current_wp_indices[Uidx] = result['wp_idx']
    
    return FLT_track, FLT_conditions, current_wp_indices