import numpy as np
from typing import Dict, List

from compute import get_climb_rate, point_in_polygon

class Thermal:
    """Classe représentant un thermique"""
    
    def __init__(self, x: float, y: float, radius: float, strength: float, duration: float, start_time: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.strength = strength  # Vitesse de montée en m/s
        self.duration = duration  # Durée en secondes
        self.start_time = start_time
        self.active = True
        
    def is_active(self, current_time: float) -> bool:
        """Vérifie si le thermique est encore actif"""
        return self.active and (current_time - self.start_time) < self.duration
    
    def get_lift_rate(self, distance_from_center: float) -> float:
        """Calcule la vitesse de montée à une distance donnée du centre"""
        
        if distance_from_center > self.radius:
            return 0.0
        
        # Profil de vitesse gaussien avec maximum au centre
        normalized_distance = distance_from_center / self.radius
        lift_factor = np.exp(-2 * normalized_distance**2)
        
        return self.strength * lift_factor

    def get_strength(self) -> float:
        """Retourne la force du thermique"""
        return self.strength

class ThermalMap:
    """Carte partagée des thermiques détectés par tous les drones"""
    
    def __init__(self):
        self.detected_thermals = {}  # {thermal_id: {'thermal': Thermal, 'detection_time': float, 'evaluated': bool, 'alt_gain': bool}}
        self.memory_duration = 600  # 10 minutes en secondes

    def add_thermal_detection(self, thermal_id: int, thermal: Thermal, detection_time: float):
        """Ajoute un thermique détecté à la carte partagée"""
        self.detected_thermals[thermal_id] = {
            'thermal': thermal,
            'detection_time': detection_time,
            'evaluated': False,
            'alt_gain': False
        }

    def get_active_thermals(self, current_time: float) -> Dict:
        """Retourne les thermiques encore en mémoire et actifs"""
        active_thermals = {}
        
        # Nettoyer les thermiques expirés de la mémoire
        expired_keys = []
        for thermal_id, data in self.detected_thermals.items():
            memory_age = current_time - data['detection_time']
            if memory_age > self.memory_duration:
                expired_keys.append(thermal_id)
        
        for key in expired_keys:
            del self.detected_thermals[key]
        
        # Retourner les thermiques actifs
        for thermal_id, data in self.detected_thermals.items():
            if data['thermal'].is_active(current_time):
                active_thermals[thermal_id] = data
                
        return active_thermals
    
    def generate_evaluation_waypoints(self, current_pos: Dict, thermal_ids: int, drone_speed, bearing) -> Dict:
        """
        Génère des waypoints d'évaluation autour de la thermique détectée.
        
        Args:
            current_pos (Dict): Position actuelle du drone
            thermal_ids (int): ID de la thermique
            drone_speed (float): Vitesse du drone [m/s]
            bearing (float): Bearing actuel du drone [radians]
            
        Returns:
            Dict: Waypoints d'évaluation {'X': [], 'Y': [], 'Z': []}
        """
        waypoints = {'X': [], 'Y': [], 'Z': []}

        if thermal_ids in self.detected_thermals:
            thermal = self.detected_thermals[thermal_ids]['thermal']
            evaluation_radius = 75  # Rayon fixe de 75m pour l'évaluation

            # Nombre de points en fonction du périmètre et de la vitesse du drone
            perimeter = 2 * np.pi * evaluation_radius
            num_waypoints = int(perimeter / drone_speed)
            
            # S'assurer qu'on a au moins 8 waypoints pour une évaluation complète
            num_waypoints = max(num_waypoints, 8)
            
            # Le bearing est déjà en radians - pas de conversion nécessaire
            start_angle_rad = bearing

            for i in range(num_waypoints):
                # Calculer les angles en radians directement
                angle_rad = start_angle_rad + (i * 2 * np.pi / num_waypoints)

                # Calculer les positions des waypoints
                x = thermal.x + (evaluation_radius * np.cos(angle_rad))
                y = thermal.y + (evaluation_radius * np.sin(angle_rad))
                waypoints['X'].append(x)
                waypoints['Y'].append(y)
                waypoints['Z'].append(current_pos['Z'])
                
            print(f"Évaluation commencée depuis bearing {np.degrees(bearing):.1f}° avec {len(waypoints['X'])} waypoints")
            
        return waypoints
    
    def change_thermal_status(self, thermal_id: int, evaluated: bool = False, alt_gain: bool = False):
        """Change le statut d'un thermique détecté"""
        if thermal_id in self.detected_thermals:
            self.detected_thermals[thermal_id]['evaluated'] = evaluated
            self.detected_thermals[thermal_id]['alt_gain'] = alt_gain
            

class ThermalGenerator:
    """Générateur de thermiques avec paramètres variables"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.thermals = dict()
        
    def generate_random_thermals(self, num_thermals: int, obstacles: Dict, current_time: float, start_id: int = 0) -> Dict:
        """Génère des thermiques aléatoires sur la carte"""
        
        for i in range(num_thermals):
            # Position aléatoire dans les limites de la carte
            x = np.random.uniform(self.params['X_lower_bound'], self.params['X_upper_bound'])
            y = np.random.uniform(self.params['Y_lower_bound'], self.params['Y_upper_bound'])
            
            # Paramètres variables
            radius = np.random.uniform(150, 300)  # Rayon entre 150 et 300m
            strength = np.random.uniform(1.5, 4.0)  # Vitesse de montée entre 1.5 et 4.0 m/s
            duration = np.random.uniform(360, 600)  # Durée entre 6 et 10 minutes
            
            # vérifier que le thermique n'est pas généré à l'intérieur d'un obstacle
            inside_obstacle = False
            for obstacle in obstacles:
                if point_in_polygon((x, y), obstacle['vertices']):
                    inside_obstacle = True
                    break
            
            if inside_obstacle:
                continue  # Regénérer ce thermique
            
            thermal_id = i + start_id
            i += 1
            
            thermal = Thermal(x, y, radius, strength, duration, current_time)
            self.thermals[thermal_id] = thermal
            
        return self.thermals

def detect_thermal_at_position(position: Dict, thermals: Dict, current_time: float) -> int:
    """
    Détecte les thermiques seulement quand le UAV est dans la zone du thermique (variation de vent)
    
    Args:
        position (Dict): Position actuelle du UAV {'X': float, 'Y': float, 'Z': float}
        thermals (Dict): Dictionnaire des thermiques actifs
        current_time (float): Temps de simulation actuel
        
    Returns:
        int: ID du thermique détecté, ou None si aucun thermique n'est détecté
    """
    for thermal_id, thermal in thermals.items():
        if not thermal.is_active(current_time):
            continue
            
        # Calculer la distance horizontale du UAV au centre du thermique
        distance = np.sqrt((position['X'] - thermal.x)**2 + (position['Y'] - thermal.y)**2)
        
        # Détection seulement si le UAV est DANS la zone du thermique
        if distance <= thermal.radius:
            # Vérifier aussi que le UAV peut sentir la variation de vent (lift_rate > 0)
            lift_rate = thermal.get_lift_rate(distance)
            if lift_rate > 0.1:  # Seuil minimum pour détecter la variation de vent
                return thermal_id

    return None


class ThermalEvaluator:
    """Évaluateur de thermiques pour les UAVs"""
    
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data
        self.evaluation_radius = 50  # Rayon d'évaluation réduit (en mètres)
        self.min_evaluation_time = 30  # Temps minimum d'évaluation en secondes
        self.max_soaring_time = 600  # Temps maximum d'exploitation en secondes (10 minutes)
        self.min_separation_distance = 50  # Distance minimale entre UAVs dans un thermique (en mètres)
        self.min_climb_rate = 0.3  # Taux de montée minimum en m/s pour considérer un thermique profitable

    def evaluate_thermal(self, thermal: 'Thermal') -> Dict:
        """
        Évalue un thermique basé sur le gain d'altitude réel observé
        
        Args:
            thermal (Thermal): Le thermique à évaluer

        Returns:
            Dict: Résultats de l'évaluation
        """
        
        # Le thermique est considéré profitable si :
        is_profitable = get_climb_rate(thermal.strength, self.max_soaring_time, self.params) >= self.min_climb_rate
        print(f"Évaluation du thermique: strength={thermal.strength:.2f} m/s -> profitable={is_profitable}")
        return is_profitable


    def check_soaring_exit_conditions(self, current_pos: Dict, thermal: 'Thermal', 
                                    soaring_start_time: float, current_time: float,
                                    other_uavs_positions: List[Dict], SOAR_WPs: List[Dict], current_soar_wp_indices: int) -> bool:
        """
        Vérifie les conditions de sortie du mode soaring.
        
        Args:
            current_pos (Dict): Position actuelle de l'UAV
            thermal (Thermal): Thermique exploité
            soaring_start_time (float): Temps de début d'exploitation
            current_time (float): Temps actuel
            other_uavs_positions (List[Dict]): Positions des autres UAVs
            
        Returns:
            bool: doit sortir du mode soaring
        """
        # Define a variable to store the exit reason
        exit_reason = None

        # 1. Vérifier la hauteur maximale
        if current_pos['Z'] >= self.params['Z_upper_bound']:
            exit_reason = "Hauteur maximale atteinte"
            print(f"Sortie du mode soaring: {exit_reason}")
            return True

        # 2. Vérifier la durée d'exploitation
        soaring_duration = current_time - soaring_start_time
        if soaring_duration >= self.max_soaring_time:
            exit_reason = f"Durée maximale d'exploitation atteinte ({self.max_soaring_time}s)"
            print(f"Sortie du mode soaring: {exit_reason}")
            return True

        # 3. Vérifier si le thermique est encore actif
        if not thermal.is_active(current_time):
            exit_reason = "Thermique non actif"
            print(f"Sortie du mode soaring: {exit_reason}")
            return True

        # 4. Vérifier la proximité avec d'autres UAVs
        for other_pos in other_uavs_positions:
            # Calculer la distance verticale entre l'UAV actuel et les autres UAVs
            distance = current_pos['Z'] - other_pos['Z']
            if distance < self.min_separation_distance:
                exit_reason = f"Distance minimale avec un autre UAV non respectée ({distance}m)"
                print(f"Sortie du mode soaring: {exit_reason}")
                return True

        # 5. Vérifier si on est encore dans le thermique
        distance_to_thermal_center = np.sqrt((current_pos['X'] - thermal.x)**2 + 
                                           (current_pos['Y'] - thermal.y)**2)
        if distance_to_thermal_center > thermal.radius:
            exit_reason = f"UAV hors du rayon du thermique ({distance_to_thermal_center:.1f}m > {thermal.radius}m)"
            print(f"Sortie du mode soaring: {exit_reason}")
            return True

        #6. vérifier si on a atteint la cible de soaring ou la liste de waypoints est vide
        if current_soar_wp_indices >= len(SOAR_WPs['X']) or not SOAR_WPs:
            exit_reason = "Waypoints de soaring terminés"
            print(f"Sortie du mode soaring: {exit_reason}")
            return True

        return False
        # 1. Vérifier la hauteur maximale
        if current_pos['Z'] >= self.params['Z_upper_bound']:
            return True
        
        # 2. Vérifier la durée d'exploitation
        soaring_duration = current_time - soaring_start_time
        if soaring_duration >= self.max_soaring_time:
            return True
        
        # 3. Vérifier si le thermique est encore actif
        if not thermal.is_active(current_time):
            return True
        
        # 4. Vérifier la proximité avec d'autres UAVs
        for other_pos in other_uavs_positions:
            # Calculer la distance verticale entre l'UAV actuel et les autres UAVs
            distance = current_pos['Z'] - other_pos['Z']
            if distance < self.min_separation_distance:
                return True
        
        # 5. Vérifier si on est encore dans le thermique
        distance_to_thermal_center = np.sqrt((current_pos['X'] - thermal.x)**2 + 
                                           (current_pos['Y'] - thermal.y)**2)
        if distance_to_thermal_center > thermal.radius:
            return True

        #6. vérifier si on a atteint la cible de soaring ou la liste de waypoints est vide
        if current_soar_wp_indices >= len(SOAR_WPs['X']) or not SOAR_WPs:
            return True

        return False
    
    def generate_soaring_trajectory(self, current_pos: Dict, thermal: 'Thermal', time_step: float, 
                               flight_conditions: Dict, current_bearing: float) -> Dict:
        """
        Génère une trajectoire de soaring optimale en spirale dans un thermique.

        Args:
            current_pos (Dict): Position actuelle de l'UAV {'X': float, 'Y': float, 'Z': float}
            thermal (Thermal): Thermique exploité
            time_step (float): Pas de temps
            flight_conditions (Dict): Conditions de vol actuelles

        Returns:
            Dict: Trajectoire de soaring {'X': [], 'Y': [], 'Z': [], 'bearing': [], 'bank_angle': []}
        """
        # Importer la fonction de calcul des paramètres optimaux
        from compute import calculate_optimal_soaring_parameters

        # Calculer les paramètres optimaux de soaring
        soaring_params = calculate_optimal_soaring_parameters(self.UAV_data, thermal, flight_conditions)

        # Extraire les paramètres optimaux (tous en radians)
        optimal_radius = soaring_params['optimal_radius']
        optimal_bank_angle_rad = soaring_params['optimal_bank_angle']  # En radians

        # Limiter le rayon optimal pour rester dans le thermique
        max_allowed_radius = thermal.radius * 0.8
        spiral_radius = min(optimal_radius, max_allowed_radius, 50)

        # Utiliser l'angle d'inclinaison optimal calculé (en radians)
        target_bank_angle_rad = optimal_bank_angle_rad

        # Calculer la vitesse angulaire basée sur la vitesse et le rayon
        airspeed = flight_conditions['airspeed']
        angular_velocity = airspeed / spiral_radius  # rad/s

        # Limiter la vitesse angulaire pour éviter des virages trop serrés
        max_angular_velocity = 1  # rad/s
        angular_velocity = min(angular_velocity, max_angular_velocity)

        # Nombre de points pour la trajectoire 
        num_points = 150

        # Position du centre du thermique
        thermal_center_x = thermal.x
        thermal_center_y = thermal.y

        # Utiliser le bearing actuel (en radians)
        current_angle = current_bearing

        print(f"Trajectoire de soaring commencée avec bearing actuel: {np.degrees(current_bearing):.1f}°")

        # Calculer la distance actuelle au centre
        current_distance = np.sqrt((current_pos['X'] - thermal_center_x)**2 + 
                                  (current_pos['Y'] - thermal_center_y)**2)
        
        # Définir le rayon de spirale sûr
        max_safe_radius = thermal.radius * 0.5
        target_radius = min(optimal_radius, max_safe_radius)
        
        # Stratégie de convergence vers optimal_radius
        if current_distance > target_radius * 1.2:
            spiral_radius = target_radius
            convergence_factor = 0.3
            radius_step = (current_distance - target_radius) / (num_points * convergence_factor)
        else:
            spiral_radius = target_radius
            radius_step = 0
   
        # Initialiser la trajectoire
        trajectory = {
            'X': [],
            'Y': [],
            'Z': [],
            'bearing': [],
            'bank_angle': [],
            'optimal_radius': []
        }

        current_altitude = current_pos['Z']
        current_radius = current_distance
        initial_bank_angle_rad = flight_conditions.get('bank_angle', 0.0)  # En radians

        # Calculer le temps de soaring cumulé
        soaring_time = 0.0

        for i in range(num_points):
            # Calculer l'angle pour ce point (en radians)
            angle = current_angle - (i * angular_velocity * time_step)

            # Ajuster le rayon progressivement
            if abs(radius_step) > 0:
                if i < num_points * convergence_factor:
                    if radius_step > 0:
                        current_radius = max(target_radius, current_distance - (i * radius_step))
                    else:
                        current_radius = min(target_radius, current_distance + (i * abs(radius_step)))
                else:
                    current_radius = target_radius
            else:
                current_radius = target_radius

            # S'assurer qu'on reste dans les limites de sécurité
            current_radius = min(current_radius, max_safe_radius)
            current_radius = max(current_radius, thermal.radius * 0.3)

            # Position en spirale
            x = thermal_center_x + current_radius * np.cos(angle)
            y = thermal_center_y + current_radius * np.sin(angle)

            distance_from_center = np.sqrt((x - thermal_center_x)**2 + (y - thermal_center_y)**2)

            # Incrémenter le temps de soaring
            soaring_time += time_step

            # Calculer le lift rate du thermique
            thermal_lift_rate = thermal.get_lift_rate(distance_from_center)
            effective_lift_rate = thermal_lift_rate

            # Utiliser l'angle d'inclinaison optimal avec transition progressive (en radians)
            if i < num_points * 0.1:
                transition_weight = i / (num_points * 0.1)
                bank_angle_rad = initial_bank_angle_rad * (1 - transition_weight) + target_bank_angle_rad * transition_weight
            else:
                bank_angle_rad = target_bank_angle_rad

            # Calculer la nouvelle altitude
            bank_factor = 1 / np.cos(bank_angle_rad)  # bank_angle_rad en radians
            adjusted_lift_rate = effective_lift_rate / bank_factor

            # Taux de chute augmenté en virage
            induced_sink = 0.5 * bank_factor - 0.5
            vertical_speed = adjusted_lift_rate - induced_sink

            altitude_change = vertical_speed * time_step
            current_altitude += altitude_change

            # Limiter l'altitude
            current_altitude = min(current_altitude, self.params['Z_upper_bound'])
            current_altitude = max(current_altitude, self.params['Z_lower_bound'])

            # Calculer le bearing (direction de vol) - tangent à la spirale (en radians)
            bearing = angle - np.pi/2

            # Normaliser le bearing dans [0, 2π]
            bearing = (bearing + 2 * np.pi) % (2 * np.pi)

            # Ajouter les points à la trajectoire (angles en radians)
            trajectory['X'].append(x)
            trajectory['Y'].append(y)
            trajectory['Z'].append(current_altitude)
            trajectory['bearing'].append(bearing)
            trajectory['bank_angle'].append(bank_angle_rad)  # Stocker en radians
            trajectory['optimal_radius'].append(optimal_radius)

        print(f"Trajectoire générée: {len(trajectory['X'])} points, "
              f"bank_angle optimal: {np.degrees(target_bank_angle_rad):.1f}°, "
              f"rayon de virage optimal: {optimal_radius:.1f}m, "
              f"vitesse optimale: {soaring_params['optimal_speed']:.1f}m/s")

        return trajectory
