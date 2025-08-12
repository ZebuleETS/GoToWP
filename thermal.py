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
    
    def generate_evaluation_waypoints(self, current_pos: Dict, thermal_ids: int) -> Dict:
        """Génère des waypoints d'évaluation autour de la thermique détectée"""
        waypoints = {'X': [], 'Y': [], 'Z': []}

        if thermal_ids in self.detected_thermals:
            thermal = self.detected_thermals[thermal_ids]['thermal']
            evaluation_radius = 25  # Rayon fixe de 25m pour l'évaluation

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
        self.thermals = dict()
        
    def generate_random_thermals(self, num_thermals: int = 3, current_time: float = 0) -> Dict:
        """Génère des thermiques aléatoires sur la carte"""
        
        for i in range(num_thermals):
            # Position aléatoire dans les limites de la carte
            x = np.random.uniform(self.params['X_lower_bound'], self.params['X_upper_bound'])
            y = np.random.uniform(self.params['Y_lower_bound'], self.params['Y_upper_bound'])
            
            # Paramètres variables
            radius = np.random.uniform(150, 300)  # Rayon entre 150 et 300m
            strength = np.random.uniform(1.5, 4.0)  # Vitesse de montée entre 1.5 et 4.0 m/s
            duration = 6000  # 10 minutes
            
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
        self.min_altitude_gain = 20.0
        self.min_climb_rate = 0.5  # Taux de montée minimum en m/s pour considérer un thermique profitable
    
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
            return {
                'positions': [], 
                'altitudes': [], 
                'altitude_changes': [],
                'total_altitude_gain': 0.0,
                'avg_climb_rate': 0.0,
                'evaluation_duration': 0.0
            }

        # Prendre les derniers points correspondant à l'évaluation
        start_idx = max(0, len(flight_track['X']) - num_evaluation_points)
        
        # Altitude de départ et finale de l'évaluation
        start_altitude = flight_track['Z'][start_idx] if start_idx < len(flight_track['Z']) else 0
        end_altitude = flight_track['Z'][-1] if flight_track['Z'] else 0

        evaluation_data = {
            'positions': [],
            'altitudes': flight_track['Z'][start_idx:],
            'altitude_changes': [],
            'start_altitude': start_altitude,
            'end_altitude': end_altitude,
            'total_altitude_gain': max(0, end_altitude - start_altitude),  # Gain net positif seulement
            'evaluation_duration': num_evaluation_points  # En supposant 1 point par seconde
        }
        print("Données d'évaluation extraites :", evaluation_data)

        # Calculer les changements d'altitude progressifs
        positive_gains = []
        for i in range(start_idx + 1, len(flight_track['Z'])):
            altitude_change = flight_track['Z'][i] - flight_track['Z'][i-1]
            evaluation_data['altitude_changes'].append(altitude_change)
            # Comptabiliser seulement les gains positifs
            if altitude_change > 0:
                positive_gains.append(altitude_change)

        # Calculer le taux de montée moyen
        if evaluation_data['evaluation_duration'] > 0:
            evaluation_data['avg_climb_rate'] = evaluation_data['total_altitude_gain'] / evaluation_data['evaluation_duration']
        else:
            evaluation_data['avg_climb_rate'] = 0.0

        # Extraire les positions
        for i in range(start_idx, len(flight_track['X'])):
            evaluation_data['positions'].append({
                'X': flight_track['X'][i],
                'Y': flight_track['Y'][i],
                'Z': flight_track['Z'][i]
            })

        # Ajouter des métriques supplémentaires
        evaluation_data['positive_altitude_gains'] = positive_gains
        evaluation_data['total_positive_gain'] = sum(positive_gains)
        evaluation_data['percentage_climbing'] = len(positive_gains) / len(evaluation_data['altitude_changes']) * 100 if evaluation_data['altitude_changes'] else 0

        return evaluation_data

    def evaluate_thermal(self, flight_track, num_evaluation_points: int) -> Dict:
        """
        Évalue un thermique basé sur le gain d'altitude réel observé
        
        Args:
            flight_track (dict): Historique de vol de l'UAV
            num_evaluation_points (int): Nombre de points d'évaluation
            
        Returns:
            bool: True si le thermique est profitable, False sinon
        if num_evaluation_points <= 0:
            return {'X': [], 'Y': [], 'Z': []}
        """
        # Extraire les données d'évaluation avec gain d'altitude
        evaluation_data = self.extract_evaluation_data(flight_track, num_evaluation_points)
        
        # Critères d'évaluation basés sur le gain d'altitude
        total_gain = evaluation_data['total_altitude_gain']
        avg_climb_rate = evaluation_data['avg_climb_rate']
        percentage_climbing = evaluation_data['percentage_climbing']
        
        # Le thermique est considéré profitable si :
        # 1. Le gain total d'altitude dépasse le minimum requis
        gain_criterion = total_gain >= self.min_altitude_gain
        
        # 2. Le taux de montée moyen est suffisant
        climb_rate_criterion = avg_climb_rate >= self.min_climb_rate
        
        # 3. Au moins 60% du temps était passé en montée
        climbing_time_criterion = percentage_climbing >= 60.0
        
        # Log des résultats d'évaluation (pour debug)
        print(f"Évaluation thermique - Gain total: {total_gain:.1f}m, "
              f"Taux moyen: {avg_climb_rate:.2f}m/s, "
              f"Temps montée: {percentage_climbing:.1f}%")
        print(f"Critères - Gain: {gain_criterion}, Taux: {climb_rate_criterion}, "
              f"Temps: {climbing_time_criterion}")
        
        # Le thermique est profitable si au moins 2 des 3 critères sont remplis
        criteria_met = sum([gain_criterion, climb_rate_criterion, climbing_time_criterion])
        is_profitable = criteria_met >= 2
        
        return is_profitable


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
        # Adapter le radius en fonction de la puissance du thermique (strength)
        # Plus le thermique est puissant, plus on utilise un petit rayon de spirale
        strength_factor = thermal.strength / 4.0  # Normaliser par rapport à la force max typique (4.0 m/s)
        
        # Calculer le rayon de spirale optimal basé sur la puissance (relation inverse)
        max_radius_ratio = 0.7  # Rayon maximum (70% du rayon du thermique pour thermiques faibles)
        min_radius_ratio = 0.3  # Rayon minimum (30% du rayon du thermique pour thermiques puissants)
        spiral_radius_ratio = max_radius_ratio - (strength_factor * (max_radius_ratio - min_radius_ratio))
        
        spiral_radius = thermal.radius * spiral_radius_ratio
        
        # S'assurer que le rayon de spirale reste dans des limites pratiques
        spiral_radius = max(15.0, min(spiral_radius, 50.0))  # Entre 15m et 50m

        # Calculer le bank angle adaptatif basé sur la puissance du thermique
        # Plus le thermique est puissant, plus le bank angle peut être élevé
        min_bank_angle = 20.0  # degrés - angle minimum pour thermiques faibles
        max_bank_angle = 35.0  # degrés - angle maximum pour thermiques puissants
        bank_angle_deg = min_bank_angle + (strength_factor * (max_bank_angle - min_bank_angle))
        bank_angle_rad = np.deg2rad(bank_angle_deg)
        
        # Calculer la vitesse angulaire basée sur le bank angle et le rayon
        # ω = g * tan(φ) / v où φ est le bank angle, v est la vitesse
        g = 9.81  # accélération gravitationnelle
        min_velocity = 8.0  
        angular_velocity = (g * np.tan(bank_angle_rad)) / (min_velocity * spiral_radius) * spiral_radius
        
        # Limiter la vitesse angulaire pour éviter des virages trop serrés
        angular_velocity = min(angular_velocity, 0.3)  # Maximum 0.3 rad/s
        angular_velocity = max(angular_velocity, 0.05)  # Minimum 0.05 rad/s

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
        #lift_rate = thermal.get_lift_rate(current_time, distance_from_center)
        lift_rate = thermal.get_strength()
        # Nouvelle altitude (montée due au thermique)
        new_z = current_pos['Z'] + lift_rate * time_step
        
        return {
            'X': new_x,
            'Y': new_y,
            'Z': new_z,
            'climb_rate': lift_rate
        }
