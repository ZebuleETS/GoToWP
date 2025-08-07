import numpy as np
from typing import Dict, List

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
    
    def get_lift_rate(self, current_time: float, distance_from_center: float) -> float:
        """Calcule la vitesse de montée à une distance donnée du centre"""
        if not self.is_active(current_time):
            return 0.0
        
        if distance_from_center > self.radius:
            return 0.0
        
        # Profil de vitesse gaussien avec maximum au centre
        normalized_distance = distance_from_center / self.radius
        lift_factor = np.exp(-2 * normalized_distance**2)
        
        return self.strength * lift_factor

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
    
    def generate_evaluation_waypoints(self, current_pos: Dict, thermal_ids: int) -> Dict:
        """Génère des waypoints d'évaluation autour de la thermique détectée"""
        waypoints = {'X': [], 'Y': [], 'Z': []}

        if thermal_ids in self.detected_thermals:
            thermal = self.detected_thermals[thermal_ids]['thermal']
            evaluation_radius = thermal.radius * 0.8  # 80% du rayon pour rester dans le thermique

            for angle in np.linspace(0, 2 * np.pi, num=8, endpoint=False):
                x = thermal.x + (evaluation_radius * np.cos(angle))
                y = thermal.y + (evaluation_radius * np.sin(angle))
                waypoints['X'].append(x)
                waypoints['Y'].append(y)
                waypoints['Z'].append(current_pos['Z'])

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
        self.thermals = {}
        
    def generate_random_thermals(self, num_thermals: int = 3, current_time: float = 0) -> Dict:
        """Génère des thermiques aléatoires sur la carte"""
        
        for i in range(num_thermals):
            # Position aléatoire dans les limites de la carte
            x = np.random.uniform(self.params['X_lower_bound'], self.params['X_upper_bound'])
            y = np.random.uniform(self.params['Y_lower_bound'], self.params['Y_upper_bound'])
            
            # Paramètres variables
            radius = np.random.uniform(150, 300)  # Rayon entre 150 et 300m
            strength = np.random.uniform(1.5, 4.0)  # Vitesse de montée entre 1.5 et 4.0 m/s
            duration = 600  # 10 minutes
            
            thermal_id = i
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
            lift_rate = thermal.get_lift_rate(current_time, distance)
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
        self.max_soaring_time = 300  # Temps maximum d'exploitation en secondes (5 minutes)
        self.min_separation_distance = 50  # Distance minimale entre UAVs dans un thermique (en mètres)

    def extract_evaluation_data(self, flight_track, num_evaluation_points):
        """
        Extrait les données de vol pendant la phase d'évaluation.

        Args:
            flight_track (dict): Historique de vol de l'UAV
            num_evaluation_points (int): Nombre de points d'évaluation

        Returns:
            dict: Données d'évaluation (positions, altitudes, temps)
        """
        if num_evaluation_points <= 0:
            return {'positions': [], 'altitudes': [], 'altitude_changes': []}

        # Prendre les derniers points correspondant à l'évaluation
        start_idx = max(0, len(flight_track['X']) - num_evaluation_points)

        evaluation_data = {
            'positions': [],
            'altitudes': flight_track['Z'][start_idx:],
            'altitude_changes': []
        }

        # Calculer les changements d'altitude
        for i in range(start_idx + 1, len(flight_track['Z'])):
            altitude_change = flight_track['Z'][i] - flight_track['Z'][i-1]
            evaluation_data['altitude_changes'].append(altitude_change)

        # Extraire les positions
        for i in range(start_idx, len(flight_track['X'])):
            evaluation_data['positions'].append({
                'X': flight_track['X'][i],
                'Y': flight_track['Y'][i],
                'Z': flight_track['Z'][i]
            })

        return evaluation_data

    def evaluate_thermal(self, flight_track, num_evaluation_points: int) -> Dict:
        """Évalue un thermique en faisant un cercle autour"""
        if num_evaluation_points <= 0:
            return {'X': [], 'Y': [], 'Z': []}
        
        # Extraire les données d'évaluation
        evaluation_data = self.extract_evaluation_data(flight_track, num_evaluation_points)

        # Calculer le gain d'altitude total et moyen
        total_altitude_gain = sum(max(0, change) for change in evaluation_data['altitude_changes'])
        avg_altitude_gain_per_step = total_altitude_gain / len(evaluation_data['altitude_changes']) if evaluation_data['altitude_changes'] else 0

        # Calculer la vitesse de montée moyenne (en m/s)
        time_step = 1.0  # Assumé 1 seconde par step
        avg_climb_rate = avg_altitude_gain_per_step / time_step

        # Critères de profitabilité
        min_climb_rate = 1.0  # m/s minimum
        min_total_gain = 10.0  # m minimum

        result = (avg_climb_rate >= min_climb_rate and total_altitude_gain >= min_total_gain)

        return result

    def check_soaring_exit_conditions(self, current_pos: Dict, thermal: 'Thermal', 
                                    soaring_start_time: float, current_time: float,
                                    other_uavs_positions: List[Dict]) -> bool:
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
        
        return False
    
    def generate_soaring_trajectory(self, current_pos: Dict, thermal: 'Thermal', 
                                  current_time: float, time_step: float) -> Dict:
        """
        Génère une trajectoire de montée en spirale dans un thermique.
        
        Args:
            current_pos (Dict): Position actuelle {'X': float, 'Y': float, 'Z': float}
            thermal (Thermal): Objet thermique à exploiter
            current_time (float): Temps de simulation actuel
            time_step (float): Pas de temps de simulation
            
        Returns:
            Dict: Nouvelle position après exploitation du thermique
        """
        # Paramètres de la spirale
        spiral_radius = thermal.radius * 0.6  # 60% du rayon du thermique
        angular_velocity = 0.1  # radians par seconde (vitesse de rotation)
        
        # Calculer la distance du centre du thermique
        distance_to_center = np.sqrt((current_pos['X'] - thermal.x)**2 + 
                                   (current_pos['Y'] - thermal.y)**2)
        
        # Si trop loin du centre, se diriger vers le centre
        if distance_to_center > spiral_radius:
            direction_to_center = np.arctan2(thermal.y - current_pos['Y'], 
                                           thermal.x - current_pos['X'])
            move_distance = min(20.0, distance_to_center - spiral_radius)  # Se rapprocher progressivement
            
            new_x = current_pos['X'] + move_distance * np.cos(direction_to_center)
            new_y = current_pos['Y'] + move_distance * np.sin(direction_to_center)
        else:
            # Effectuer une spirale autour du centre
            current_angle = np.arctan2(current_pos['Y'] - thermal.y, 
                                     current_pos['X'] - thermal.x)
            new_angle = current_angle + angular_velocity * time_step
            
            new_x = thermal.x + spiral_radius * np.cos(new_angle)
            new_y = thermal.y + spiral_radius * np.sin(new_angle)
        
        # Calculer la vitesse de montée basée sur la position dans le thermique
        distance_from_center = np.sqrt((new_x - thermal.x)**2 + (new_y - thermal.y)**2)
        lift_rate = thermal.get_lift_rate(current_time, distance_from_center)
        
        # Nouvelle altitude (montée due au thermique)
        new_z = current_pos['Z'] + lift_rate * time_step
        
        return {
            'X': new_x,
            'Y': new_y,
            'Z': new_z,
            'climb_rate': lift_rate
        }
