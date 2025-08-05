import numpy as np
import time
from typing import Dict, List, Tuple
from compute import compute_distance_cartesian, get_destination_from_range_and_bearing_cartesian

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
        self.detected_thermals = {}  # {thermal_id: {'thermal': Thermal, 'detection_time': float}}
        self.memory_duration = 600  # 10 minutes en secondes

    def add_thermal_detection(self, thermal_id: int, thermal: Thermal, detection_time: float):
        """Ajoute un thermique détecté à la carte partagée"""
        self.detected_thermals[thermal_id] = {
            'thermal': thermal,
            'detection_time': detection_time
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


#pas utiliser
class ThermalEvaluator:
    """Évaluateur de thermiques pour les UAVs"""
    
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data
        self.evaluation_radius = 50  # Rayon d'évaluation réduit (en mètres)
        self.min_evaluation_time = 30  # Temps minimum d'évaluation en secondes
        
    def evaluate_thermal(self, uav_position: Dict, thermal: Thermal, current_time: float) -> Dict:
        """Évalue un thermique en faisant un cercle autour"""
        
        # Vérifier si le UAV est dans le thermique pour pouvoir l'évaluer
        distance_to_thermal = np.sqrt((uav_position['X'] - thermal.x)**2 + 
                                    (uav_position['Y'] - thermal.y)**2)
        
        # Le UAV doit être dans le thermique ou très proche pour l'évaluer
        if distance_to_thermal > thermal.radius:
            return {'can_evaluate': False, 'reason': 'not_in_thermal'}
        
        # Simuler une évaluation circulaire depuis l'intérieur du thermique
        evaluation_result = self._simulate_circular_evaluation(uav_position, thermal, current_time)
        
        return evaluation_result
    
    def _simulate_circular_evaluation(self, uav_position: Dict, thermal: Thermal, current_time: float) -> Dict:
        """Simule l'évaluation en cercle du thermique depuis l'intérieur"""
        
        # Points d'évaluation autour de la position actuelle (pas du centre du thermique)
        num_points = 8
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        total_lift = 0
        valid_points = 0
        
        for angle in angles:
            # Position d'évaluation autour de la position actuelle du UAV
            eval_x = uav_position['X'] + self.evaluation_radius * np.cos(angle)
            eval_y = uav_position['Y'] + self.evaluation_radius * np.sin(angle)
            
            # Distance du point d'évaluation au centre du thermique
            distance_to_center = np.sqrt((eval_x - thermal.x)**2 + (eval_y - thermal.y)**2)
            
            # Obtenir la vitesse de montée à ce point
            lift_rate = thermal.get_lift_rate(current_time, distance_to_center)
            
            if lift_rate > 0:
                total_lift += lift_rate
                valid_points += 1
        
        # Calculer la vitesse de montée moyenne
        avg_lift_rate = total_lift / max(valid_points, 1)
        
        # Critères d'évaluation
        min_lift_threshold = 1.0  # m/s minimum pour considérer le thermique viable
        
        if avg_lift_rate >= min_lift_threshold and valid_points >= 4:  # Au moins la moitié des points doivent être valides
            estimated_altitude_gain = avg_lift_rate * self.min_evaluation_time
            return {
                'can_evaluate': True,
                'positive': True,
                'avg_lift_rate': avg_lift_rate,
                'estimated_gain': estimated_altitude_gain,
                'thermal_center': {'X': thermal.x, 'Y': thermal.y, 'Z': uav_position['Z']},
                'thermal_radius': thermal.radius
            }
        else:
            return {
                'can_evaluate': True,
                'positive': False,
                'avg_lift_rate': avg_lift_rate,
                'reason': 'insufficient_lift'
            }

class ThermalExploiter:
    """Gère l'exploitation des thermiques par les UAVs"""
    
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data
        self.max_exploitation_time = 300  # 5 minutes maximum
        self.min_separation_distance = 100  # Distance minimale entre UAVs dans un thermique
        
    def can_exploit_thermal(self, uav_id: int, thermal_center: Dict, other_uavs_positions: List[Dict]) -> bool:
        """Vérifie si un UAV peut exploiter un thermique"""
        
        # Vérifier la proximité avec d'autres UAVs
        for other_pos in other_uavs_positions:
            distance = compute_distance_cartesian(thermal_center, other_pos)[0]
            if distance < self.min_separation_distance:
                return False
        
        return True
    
    def generate_thermal_exploitation_path(self, uav_position: Dict, thermal_center: Dict, 
                                         thermal_radius: float, current_time: float) -> Dict:
        """Génère une trajectoire en spirale pour exploiter le thermique"""
        
        # Paramètres de la spirale
        num_turns = 3
        points_per_turn = 20
        total_points = num_turns * points_per_turn
        
        # Rayon de la spirale (plus petit que le rayon du thermique)
        spiral_radius = min(thermal_radius * 0.7, 80)  # Maximum 80m de rayon
        
        # Génération des points de la spirale
        angles = np.linspace(0, num_turns * 2 * np.pi, total_points)
        
        x_points = []
        y_points = []
        z_points = []
        
        for i, angle in enumerate(angles):
            # Rayon progressif de la spirale
            current_radius = spiral_radius * (1 - i / total_points * 0.3)  # Spirale qui se resserre
            
            x = thermal_center['X'] + current_radius * np.cos(angle)
            y = thermal_center['Y'] + current_radius * np.sin(angle)
            
            # Altitude progressive (montée dans le thermique)
            altitude_gain_per_point = 2.0  # 2m de gain par point
            z = uav_position['Z'] + i * altitude_gain_per_point
            z = min(z, self.params['Z_upper_bound'])  # Limiter à l'altitude max
            
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
        
        return {
            'X': x_points,
            'Y': y_points,
            'Z': z_points,
            'type': 'thermal_exploitation',
            'start_time': current_time,
            'thermal_center': thermal_center
        }