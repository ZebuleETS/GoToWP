"""
Configuration des paramètres PX4 pour correspondre exactement à votre planeur
Basé sur les paramètres UAV_data de votre simulation
"""

import asyncio
import numpy as np

# Vos paramètres de simulation
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
params['safe_distance'] = 30.0
params['horizon_length'] = 100.0


def calculate_flight_parameters(UAV_data, params):
    """
    Calculer les paramètres de vol optimaux basés sur vos données UAV
    """
    # Constantes atmosphériques
    ACC_SEA_LEVEL = 9.80665
    T_SEA_LEVEL = 288.15
    RHO_SEA_LEVEL = 1.225
    MEAN_EARTH_RADIUS = 6371009
    TROPO_LAPSE_RATE = -0.0065
    R = 287.058
    
    # Calculer la gravité à l'altitude de travail
    grav_accel = ACC_SEA_LEVEL * (MEAN_EARTH_RADIUS / (MEAN_EARTH_RADIUS + params['working_floor']))
    
    # Calculer la densité de l'air à l'altitude de travail
    T_fin = T_SEA_LEVEL + TROPO_LAPSE_RATE * params['working_floor']
    air_density = RHO_SEA_LEVEL * (T_fin / T_SEA_LEVEL)**(-grav_accel / (TROPO_LAPSE_RATE * R) - 1)
    
    # Calculer la vitesse de décrochage (stall speed)
    # V_stall = sqrt(2 * Weight * g / (rho * S * CL_max))
    # En utilisant CL_max ≈ 1.5 pour un planeur typique
    CL_max = 1.5
    weight = UAV_data['empty_weight'] * grav_accel
    V_stall = np.sqrt(2 * weight / (air_density * UAV_data['wing_area'] * CL_max))
    
    # Calculer la vitesse optimale de croisière (L/D max)
    # Pour un planeur, vitesse de croisière ≈ 1.3 * V_stall à 1.5 * V_stall
    V_cruise_optimal = 1.4 * V_stall
    
    # Calculer le finesse max (L/D) approximatif
    # L/D_max ≈ 0.5 * sqrt(π * AR * e / CD0)
    LD_max = 0.5 * np.sqrt(np.pi * UAV_data['wing_aspect_ratio'] * 
                           UAV_data['oswald_eff_ratio'] / UAV_data['zero_lift_drag'])
    
    # Calculer le taux de chute minimal
    # V_sink_min ≈ sqrt((2 * Weight / (rho * S)) * sqrt(CD0 / (3 * π * AR * e)))
    V_sink_min = np.sqrt((2 * weight / (air_density * UAV_data['wing_area'])) * 
                         np.sqrt(UAV_data['zero_lift_drag'] / (3 * np.pi * 
                         UAV_data['wing_aspect_ratio'] * UAV_data['oswald_eff_ratio'])))
    
    # Calculer le rayon de virage minimum
    # r_min = V^2 / (g * tan(bank_angle_max))
    # Avec bank_angle_max ≈ 45° pour un planeur
    bank_angle_max_rad = np.radians(45)
    turn_radius_min = (V_cruise_optimal**2) / (grav_accel * np.tan(bank_angle_max_rad))
    
    # Calculer le taux de virage
    # turn_rate = g * tan(bank_angle) / V
    turn_rate_max = (grav_accel * np.tan(bank_angle_max_rad)) / V_cruise_optimal
    turn_rate_max_deg = np.degrees(turn_rate_max)
    
    results = {
        'air_density': air_density,
        'gravity': grav_accel,
        'stall_speed': V_stall,
        'cruise_speed_optimal': V_cruise_optimal,
        'LD_max': LD_max,
        'sink_rate_min': V_sink_min,
        'turn_radius_min': turn_radius_min,
        'turn_rate_max': turn_rate_max_deg,
        'bank_angle_max': 45.0
    }
    
    return results


def generate_px4_parameters(UAV_data, params):
    """
    Générer les paramètres PX4 optimaux basés sur vos données
    """
    # Calculer les paramètres de vol
    flight_params = calculate_flight_parameters(UAV_data, params)
    
    print("="*70)
    print("PARAMÈTRES CALCULÉS POUR VOTRE PLANEUR")
    print("="*70)
    print(f"Densité de l'air @ {params['working_floor']}m: {flight_params['air_density']:.4f} kg/m³")
    print(f"Gravité @ {params['working_floor']}m: {flight_params['gravity']:.4f} m/s²")
    print(f"Vitesse de décrochage calculée: {flight_params['stall_speed']:.2f} m/s")
    print(f"Vitesse de croisière optimale: {flight_params['cruise_speed_optimal']:.2f} m/s")
    print(f"Finesse maximale (L/D): {flight_params['LD_max']:.2f}")
    print(f"Taux de chute minimal: {flight_params['sink_rate_min']:.3f} m/s")
    print(f"Rayon de virage minimal: {flight_params['turn_radius_min']:.1f} m")
    print(f"Taux de virage maximal: {flight_params['turn_rate_max']:.2f} °/s")
    print("="*70)
    
    # Générer les paramètres PX4
    px4_params = {
        # ===== VITESSES AÉRODYNAMIQUES =====
        'FW_AIRSPD_MIN': UAV_data['min_airspeed'],           # Vitesse minimale de sécurité
        'FW_AIRSPD_MAX': UAV_data['max_airspeed'],           # Vitesse maximale
        'FW_AIRSPD_TRIM': flight_params['cruise_speed_optimal'],  # Vitesse de croisière optimale
        'FW_AIRSPD_STALL': flight_params['stall_speed'],     # Vitesse de décrochage
        
        # ===== LIMITES D'ANGLES =====
        'FW_R_LIM': flight_params['bank_angle_max'],         # Angle de roulis max (degrés)
        'FW_P_LIM_MAX': 30.0,                                # Angle de tangage max (degrés)
        'FW_P_LIM_MIN': -30.0,                               # Angle de tangage min (degrés)
        
        # ===== PARAMÈTRES DE VIRAGE =====
        'FW_R_TC': 0.4,                                      # Time constant pour roulis (s)
        'FW_P_TC': 0.4,                                      # Time constant pour tangage (s)
        
        # ===== ALTITUDE ET NAVIGATION =====
        'FW_ALT_MODE': 0,                                    # Mode altitude (0=AMSL)
        'NAV_LOITER_RAD': max(80.0, flight_params['turn_radius_min'] * 1.2),  # Rayon de loiter
        'NAV_ACC_RAD': params['safe_distance'],              # Rayon d'acceptance waypoint
        'MIS_LTRMIN_ALT': params['Z_lower_bound'],          # Altitude minimale de loiter
        
        # ===== PARAMÈTRES DE MONTÉE/DESCENTE =====
        'FW_T_CLMB_MAX': 5.0,                                # Taux de montée max (m/s)
        'FW_T_SINK_MIN': flight_params['sink_rate_min'],    # Taux de descente min (m/s)
        'FW_T_SINK_MAX': 5.0,                                # Taux de descente max (m/s)
        'FW_T_ALT_TC': 5.0,                                  # Time constant altitude (s)
        
        # ===== GÉOFENCE =====
        'GF_ACTION': 1,                                      # Action geofence (1=RTL)
        'GF_MAX_HOR_DIST': params['X_upper_bound'],         # Distance horizontale max (m)
        'GF_MAX_VER_DIST': params['Z_upper_bound'] - params['Z_lower_bound'],  # Distance verticale max (m)
        
        # ===== MODE OFFBOARD =====
        'COM_OF_LOSS_T': 5.0,                                # Timeout offboard (s)
        'COM_OBL_ACT': 1,                                    # Action si perte offboard (1=hold)
        'COM_POS_FS_DELAY': 1,                               # Délai failsafe position (s)
        
        # ===== BATTERIE =====
        'BAT_CAPACITY': UAV_data['maximum_battery_capacity'] * 1000,  # Capacité (mAh)
        'BAT_V_CHARGED': 12.6,                               # Voltage chargé (V) - ajuster selon votre batterie
        'BAT_V_EMPTY': 9.0,                                  # Voltage vide (V) - ajuster selon votre batterie
        'BAT_CRIT_THR': UAV_data['desired_reserved_battery_capacity'] / UAV_data['maximum_battery_capacity'],
        'BAT_LOW_THR': 0.3,                                  # Seuil bas (30%)
        'BAT_EMERGEN_THR': 0.1,                              # Seuil urgence (10%)
        
        # ===== PERFORMANCES =====
        'FW_SERVICE_CEIL': params['Z_upper_bound'],         # Plafond de service (m)
        'FW_THR_MAX': 1.0,                                   # Throttle max (100%)
        'FW_THR_MIN': 0.0,                                   # Throttle min (0% - planeur)
        'FW_THR_CRUISE': 0.0,                                # Throttle croisière (0% - planeur)
        
        # ===== LANDING =====
        'FW_LND_ANG': 5.0,                                   # Angle d'approche atterrissage (degrés)
        'FW_LND_FLALT': 5.0,                                 # Altitude de flare (m)
        'FW_LND_AIRSPD': UAV_data['min_airspeed'] * 1.1,    # Vitesse d'atterrissage
        
        # ===== L1 CONTROLLER (Path Following) =====
        'FW_L1_PERIOD': max(15.0, flight_params['turn_radius_min'] / 5.0),  # L1 période
        'FW_L1_DAMPING': 0.75,                               # L1 damping
        
        # ===== MISSION =====
        'MIS_TAKEOFF_ALT': params['working_floor'],         # Altitude de décollage
        'MIS_DIST_1WP': 500,                                 # Distance avant 1er WP (m)
        'MIS_DIST_WPS': 900,                                 # Distance max entre WPs (m)
    }
    
    return px4_params, flight_params


async def configure_px4_parameters():
    """
    Afficher les paramètres recommandés
    """
    px4_params, flight_params = generate_px4_parameters(UAV_data, params)
    
    print("\n" + "="*70)
    print("PARAMÈTRES PX4 RECOMMANDÉS")
    print("="*70)
    
    # Grouper par catégories
    categories = {
        'Vitesses': ['FW_AIRSPD_MIN', 'FW_AIRSPD_MAX', 'FW_AIRSPD_TRIM', 'FW_AIRSPD_STALL'],
        'Angles': ['FW_R_LIM', 'FW_P_LIM_MAX', 'FW_P_LIM_MIN'],
        'Navigation': ['NAV_LOITER_RAD', 'NAV_ACC_RAD', 'MIS_LTRMIN_ALT'],
        'Altitude': ['FW_T_CLMB_MAX', 'FW_T_SINK_MIN', 'FW_T_SINK_MAX'],
        'Géofence': ['GF_MAX_HOR_DIST', 'GF_MAX_VER_DIST'],
        'Batterie': ['BAT_CAPACITY', 'BAT_CRIT_THR', 'BAT_LOW_THR'],
    }
    
    for category, param_names in categories.items():
        print(f"\n{category}:")
        for param_name in param_names:
            if param_name in px4_params:
                value = px4_params[param_name]
                print(f"  {param_name:<20} = {value:>10.2f}")
    
    return px4_params, flight_params


def generate_px4_config_file(params_dict, flight_params, filename="px4_glider_params.txt"):
    """
    Génère un fichier de configuration PX4
    """
    config_content = f"""#!/bin/sh
#
# Configuration automatique pour planeur avec thermiques
# Généré automatiquement à partir des paramètres de simulation
#
# @name Planeur Personnalisé Soaring
# @type Fixed Wing
#

. ${{R}}etc/init.d/rc.fw_defaults

echo "Configuration planeur soaring"

# ===== CARACTÉRISTIQUES DU PLANEUR =====
# Masse: {UAV_data['empty_weight']} kg
# Surface alaire: {UAV_data['wing_area']} m²
# Allongement: {UAV_data['wing_aspect_ratio']}
# Finesse max: {flight_params['LD_max']:.1f}
# Vitesse de décrochage: {flight_params['stall_speed']:.1f} m/s
# Vitesse de croisière optimale: {flight_params['cruise_speed_optimal']:.1f} m/s

# ===== PARAMÈTRES PX4 =====
"""
    
    for param_name, param_value in params_dict.items():
        config_content += f"param set {param_name} {param_value}\n"
    
    config_content += """
# ===== MIXER ET SORTIE =====
set MIXER fw_generic_wing
set PWM_OUT 1234

# ===== MODE SOARING =====
# Note: Le soaring est géré par votre code Python
# PX4 fournit uniquement le contrôle bas-niveau

echo "Configuration planeur terminée"
"""
    
    with open(filename, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ Fichier de configuration généré: {filename}")
    print(f"   Pour l'utiliser:")
    print(f"   1. Copiez-le dans PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/")
    print(f"   2. Nommez-le avec un numéro unique (ex: 2106_glider_soaring)")
    print(f"   3. Redémarrez PX4 avec: make px4_sitl gazebo_plane")


async def set_parameters_via_mavlink(params_dict):
    """
    Définir les paramètres via MAVLink (nécessite pymavlink)
    """
    try:
        from pymavlink import mavutil
        
        print("\n" + "="*70)
        print("Configuration via MAVLink")
        print("="*70)
        
        print("Connexion à PX4...")
        master = mavutil.mavlink_connection('udp:127.0.0.1:14540')
        master.wait_heartbeat()
        print("✓ Heartbeat reçu")
        
        print("\nEnvoi des paramètres...")
        success_count = 0
        fail_count = 0
        
        for param_name, param_value in params_dict.items():
            try:
                # Envoyer la commande de paramètre
                master.mav.param_set_send(
                    master.target_system,
                    master.target_component,
                    param_name.encode('utf-8'),
                    float(param_value),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                )
                
                # Attendre la confirmation
                msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)
                if msg and msg.param_id.decode('utf-8').rstrip('\x00') == param_name:
                    print(f"  ✓ {param_name:<20} = {param_value:>10.2f}")
                    success_count += 1
                else:
                    print(f"  ✗ {param_name:<20} - Timeout ou erreur")
                    fail_count += 1
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"  ✗ {param_name} - Erreur: {e}")
                fail_count += 1
        
        print(f"\n✅ Configuration terminée: {success_count} réussis, {fail_count} échecs")
        
        if fail_count > 0:
            print("⚠️  Certains paramètres n'ont pas pu être configurés")
            print("   Essayez de les configurer manuellement via QGroundControl")
        
    except ImportError:
        print("\n⚠️  pymavlink non installé")
        print("   Installation: pip install pymavlink")
    except Exception as e:
        print(f"\n✗ Erreur MAVLink: {e}")


async def verify_parameters(params_to_check):
    """
    Vérifier que les paramètres sont correctement configurés
    """
    try:
        from pymavlink import mavutil
        
        print("\n" + "="*70)
        print("VÉRIFICATION DES PARAMÈTRES")
        print("="*70)
        
        master = mavutil.mavlink_connection('udp:127.0.0.1:14540')
        master.wait_heartbeat()
        
        print("\nVérification en cours...")
        correct_count = 0
        incorrect_count = 0
        
        for param_name, expected_value in params_to_check.items():
            # Demander le paramètre spécifique
            master.mav.param_request_read_send(
                master.target_system,
                master.target_component,
                param_name.encode('utf-8'),
                -1
            )
            
            msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)
            if msg:
                actual_value = msg.param_value
                param_id = msg.param_id.decode('utf-8').rstrip('\x00')
                
                if param_id == param_name:
                    diff = abs(actual_value - expected_value)
                    tolerance = max(0.01, abs(expected_value) * 0.01)  # 1% de tolérance
                    
                    if diff < tolerance:
                        print(f"  ✓ {param_name:<20} = {actual_value:>10.2f} (attendu: {expected_value:.2f})")
                        correct_count += 1
                    else:
                        print(f"  ✗ {param_name:<20} = {actual_value:>10.2f} (attendu: {expected_value:.2f}) DIFFÉRENT!")
                        incorrect_count += 1
            else:
                print(f"  ? {param_name:<20} - Pas de réponse")
                incorrect_count += 1
            
            await asyncio.sleep(0.05)
        
        print(f"\n{'='*70}")
        print(f"Résultat: {correct_count}/{len(params_to_check)} paramètres corrects")
        if incorrect_count > 0:
            print(f"⚠️  {incorrect_count} paramètres nécessitent une attention")
        else:
            print("✅ Tous les paramètres sont correctement configurés!")
        
    except Exception as e:
        print(f"\n✗ Erreur lors de la vérification: {e}")


async def main():
    """
    Fonction principale pour configurer PX4
    """
    print("="*70)
    print("CONFIGURATION PX4 POUR VOTRE SIMULATION DE PLANEUR")
    print("="*70)
    
    # Générer les paramètres
    px4_params, flight_params = await configure_px4_parameters()
    
    # Générer le fichier de config
    print("\n" + "="*70)
    print("GÉNÉRATION DU FICHIER DE CONFIGURATION")
    print("="*70)
    generate_px4_config_file(px4_params, flight_params)
    
    # Demander si on applique maintenant
    print("\n" + "="*70)
    print("APPLICATION DES PARAMÈTRES")
    print("="*70)
    print("Voulez-vous appliquer ces paramètres maintenant via MAVLink?")
    print("(PX4 SITL doit être en cours d'exécution)")
    
    response = input("\nAppliquer maintenant? [y/N]: ").strip().lower()
    if response == 'y':
        await set_parameters_via_mavlink(px4_params)
        
        # Vérifier
        verify = input("\nVérifier les paramètres? [y/N]: ").strip().lower()
        if verify == 'y':
            await verify_parameters(px4_params)
    else:
        print("\n⚠️  Paramètres non appliqués")
        print("   Options pour les appliquer plus tard:")
        print("   1. Relancer ce script quand PX4 est en cours d'exécution")
        print("   2. Utiliser le fichier de config généré")
        print("   3. Configurer manuellement via QGroundControl")
    
    print("\n" + "="*70)
    print("✅ CONFIGURATION TERMINÉE")
    print("="*70)
    print("\nProchaines étapes:")
    print("1. Lancer PX4 SITL: make px4_sitl gazebo_plane")
    print("2. Lancer votre simulation: python3 example_px4_integrated.py")
    print("3. Monitorer avec QGroundControl (optionnel)")


if __name__ == "__main__":
    asyncio.run(main())