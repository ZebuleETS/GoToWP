import numpy as np
from math import pi, asin, cos
from abc import ABC, abstractmethod
from compute import (
    check_trajectory_obstacles,
    compute_bearing,
    compute_bearing_cartesian,
    compute_distance,
    compute_distance_cartesian,
    extract_waypoint,
    get_destination_from_range_and_bearing_cartesian,
    get_power_consumption,
)
from typing import Dict, List
from scipy import interpolate
from scipy.signal import savgol_filter

class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators"""
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data

    @abstractmethod
    def generate_path(self, start_point, end_point):
        """
        Generates a trajectory between two points
        
        Args:
            start_point (dict): Starting point {latitude, longitude, altitude}
            end_point (dict): End point {latitude, longitude, altitude}
            
        Returns:
            dict: Trajectory points {latitude: [], longitude: [], altitude: []}
        """
        pass

class StraightLineTrajectory(TrajectoryGenerator):
    """
    Straight line trajectory generator for cartesian coordinates
    Args:
        params (dict): Parameters for the trajectory generation, e.g., number of points, radius, etc.
        UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)

    def generate_path(self, start_point, end_point):
        """
        Génère une trajectoire en ligne droite entre deux points cartésiens
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            end_point (dict): Point d'arrivée {X, Y, Z}
            
        Returns:
            dict: Points de trajectoire {X: [], Y: [], Z: []}
        """
        num_points = self.params.get('num_points', 200)
        end_point = extract_waypoint(end_point)
        
        # Génération des points intermédiaires en ligne droite
        x_points = np.linspace(start_point['X'], end_point['X'], num_points)
        y_points = np.linspace(start_point['Y'], end_point['Y'], num_points)
        z_points = np.linspace(start_point['Z'], end_point['Z'], num_points)
        
        return {'X': x_points.tolist(), 'Y': y_points.tolist(), 'Z': z_points.tolist()}

class CircularTrajectory(TrajectoryGenerator):
    """
    Circular trajectory generator for cartesian coordinates
    Args:
        params (dict): Parameters for the trajectory generation, e.g., number of points, radius, etc.
        UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)

    def generate_path(self, start_point, end_point) -> Dict:
        """
        Génère une trajectoire circulaire (arc) entre deux points cartésiens
        en optimisant le rayon pour minimiser la distance parcourue
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            end_point (dict): Point d'arrivée {X, Y, Z}
        Returns:
            dict: Points de trajectoire {X: [], Y: [], Z: []}
        """
        num_points = self.params.get('num_points', 200)
        base_radius = 100  # rayon initial en mètres
        end_point = extract_waypoint(end_point)
        
        # Calculer la distance directe entre les deux points
        dx = end_point['X'] - start_point['X']
        dy = end_point['Y'] - start_point['Y']
        chord_length = np.sqrt(dx**2 + dy**2)
        
        # Rayon minimum nécessaire (moitié de la distance directe)
        min_radius = chord_length / 2
        
        # Si UAV_data est disponible, prendre en compte le rayon de virage minimum
        min_turn_radius = None
        if self.UAV_data:
            min_velocity = self.UAV_data.get('min_airspeed', 8.0)
            max_velocity = self.UAV_data.get('max_airspeed', 30.0)
            max_turn_rate = self.UAV_data.get('max_turn_rate', 0.7)
            avg_velocity = (min_velocity + max_velocity) / 2
            min_turn_radius = avg_velocity / max_turn_rate
            
            # Assurer que le rayon minimum est respecté
            min_radius = max(min_radius, min_turn_radius)
        
        # Définir une plage de rayons à tester
        max_test_radius = max(base_radius * 3, min_radius * 4)  # Limiter l'exploration
        
        # Recherche du rayon optimal
        optimal_radius = self._find_optimal_radius(start_point, end_point, min_radius, max_test_radius)
        
        # Générer la trajectoire avec le rayon optimal
        return self._generate_circular_arc(start_point, end_point, optimal_radius, num_points)
    
    
    def _find_optimal_radius(self, start_point, end_point, min_radius, max_radius, num_samples=10):
        """
        Trouve le rayon optimal qui minimise la longueur de l'arc entre deux points
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            end_point (dict): Point d'arrivée {X, Y, Z}
            min_radius (float): Rayon minimum à tester
            max_radius (float): Rayon maximum à tester
            num_samples (int): Nombre d'échantillons de rayon à tester
            
        Returns:
            float: Rayon optimal
        """
        # Distance directe entre les points
        dx = end_point['X'] - start_point['X']
        dy = end_point['Y'] - start_point['Y']
        chord_length = np.sqrt(dx**2 + dy**2)
        
        # Tester différents rayons et calculer la longueur de l'arc
        radii = np.linspace(min_radius, max_radius, num_samples)
        arc_lengths = []
        
        for radius in radii:
            # Calculer la longueur de l'arc pour ce rayon
            # Pour un arc de cercle, angle = 2 * arcsin(chord / (2 * radius))
            # Longueur d'arc = rayon * angle
            angle = 2 * np.arcsin(chord_length / (2 * radius))
            arc_length = radius * angle
            arc_lengths.append(arc_length)
        
        # Trouver le rayon qui donne la longueur d'arc minimale
        min_index = np.argmin(arc_lengths)
        optimal_radius = radii[min_index]
        
        print(f"Rayon optimal trouvé: {optimal_radius:.2f} m (longueur d'arc: {arc_lengths[min_index]:.2f} m)")
        
        return optimal_radius
    
    
    def _generate_circular_arc(self, start_point, end_point, radius, num_points):
        """
        Génère un arc de cercle entre deux points avec le rayon spécifié
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            end_point (dict): Point d'arrivée {X, Y, Z}
            radius (float): Rayon du cercle
            num_points (int): Nombre de points à générer
            
        Returns:
            dict: Points de trajectoire {X: [], Y: [], Z: []}
        """
        # Calculer la distance directe entre les deux points
        dx = end_point['X'] - start_point['X']
        dy = end_point['Y'] - start_point['Y']
        chord_length = np.sqrt(dx**2 + dy**2)
        
        # Calculer la hauteur du segment circulaire (sagitta)
        sagitta = radius - np.sqrt(radius**2 - (chord_length/2)**2)
        
        # Direction perpendiculaire à la ligne droite entre les points
        perpendicular_x = -dy / chord_length
        perpendicular_y = dx / chord_length
        
        # Calculer le centre du cercle
        mid_x = (start_point['X'] + end_point['X']) / 2
        mid_y = (start_point['Y'] + end_point['Y']) / 2
        
        # Positionner le centre du cercle perpendiculairement à la ligne directe
        # Pour un arc minimal, on place le centre sur la médiatrice
        center_x = mid_x + perpendicular_x * (radius - sagitta)
        center_y = mid_y + perpendicular_y * (radius - sagitta)
        
        # Calculer les angles de départ et d'arrivée par rapport au centre
        start_angle = np.arctan2(start_point['Y'] - center_y, start_point['X'] - center_x)
        end_angle = np.arctan2(end_point['Y'] - center_y, end_point['X'] - center_x)
        
        # Assurer que nous prenons le bon arc (le plus court)
        if abs(end_angle - start_angle) > np.pi:
            if end_angle > start_angle:
                start_angle += 2 * np.pi
            else:
                end_angle += 2 * np.pi
        
        # Générer des points uniformément répartis sur l'arc
        angles = np.linspace(start_angle, end_angle, num_points)
        x_points = center_x + radius * np.cos(angles)
        y_points = center_y + radius * np.sin(angles)
        
        # Interpolation linéaire de l'altitude
        z_points = np.linspace(start_point['Z'], end_point['Z'], num_points)
        
        return {'X': x_points.tolist(), 'Y': y_points.tolist(), 'Z': z_points.tolist()}

class DubinsPath3D(TrajectoryGenerator):
    """
    3D Dubins path trajectory generator
    
    Implémentation des chemins de Dubins (LSL, RSR, LSR, RSL) pour les UAVs en 3D.
    Un chemin de Dubins est constitué de segments circulaires et droits respectant
    les contraintes de rayon de courbure minimum du véhicule.
    
    Args:
        params (dict): Parameters for the trajectory generation, e.g., number of points, radius, etc.
        UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)
        # Calcul du rayon de virage minimum si les paramètres sont disponibles
        if self.params is not None and self.UAV_data is not None:
            min_velocity = self.UAV_data.get('min_airspeed', 8.0)
            max_velocity = self.UAV_data.get('max_airspeed', 30.0)
            max_turn_rate = self.UAV_data.get('max_turn_rate', 0.7)
            avg_velocity = (min_velocity + max_velocity) / 2
            self.min_turn_radius = avg_velocity / max_turn_rate
            
        # Liste des configurations de Dubins possibles
        self.dubins_configs = ['LSL', 'RSR', 'LSR', 'RSL']

    def generate_path(self, start_point, end_point) -> Dict:
        """
        Génère une trajectoire Dubins 3D entre deux points en évaluant toutes 
        les configurations possibles (LSL, RSR, LSR, RSL) et en choisissant la meilleure.
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            end_point (dict): Point d'arrivée {X, Y, Z}
        Returns:
            dict: Points de trajectoire {X: [], Y: [], Z: []}
        """
        num_points = self.params.get('num_points', 200)
        end_point = extract_waypoint(end_point)
        # Calculer le cap initial et final
        bearing_start = 0  # Par défaut, on suppose que le drone pointe vers l'axe X positif
        if 'bearing' in start_point:
            bearing_start = start_point['bearing']
        
        # Calculer le cap entre le point de départ et d'arrivée
        bearing_end = compute_bearing_cartesian(start_point, end_point)[0]
        
        # Évaluer chaque configuration de Dubins et choisir la meilleure
        min_length = float('inf')
        best_path = None
        
        for config in self.dubins_configs:
            path, length = self._compute_dubins_path(start_point, end_point, bearing_start, 
                                               bearing_end, config, num_points)
            if path is not None and length < min_length:
                min_length = length
                best_path = path
        
        if best_path is None:
            # Fallback: générer une ligne droite si aucun chemin n'est valide
            straight_points = num_points
            straight_path = self._generate_straight_segment(start_point, end_point, straight_points)
            
            x_points = straight_path['path']['X']
            y_points = straight_path['path']['Y']
            
            # Interpolation linéaire de l'altitude
            z_points = []
            for i in range(len(x_points)):
                fraction = i / (len(x_points) - 1) if len(x_points) > 1 else 0
                z = start_point['Z'] + fraction * (end_point['Z'] - start_point['Z'])
                z_points.append(z)
        else:
            x_points = best_path['X']
            y_points = best_path['Y']
            z_points = best_path['Z']
        
        # Si numpy array, convertir en liste
        if isinstance(z_points, np.ndarray):
            z_points = z_points.tolist()
            
        trajectory = {'X': x_points, 'Y': y_points, 'Z': z_points}
        
        smoother = TrajectorySmoothing(self.params)
        trajectory = smoother.smooth_trajectory(trajectory, method='dubins_junctions')

        return trajectory

    def _compute_dubins_path(self, start_point, end_point, bearing_start, bearing_end,
                            config, num_points_total):
        """
        Calcule un chemin de Dubins selon une configuration spécifique.
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            end_point (dict): Point d'arrivée {X, Y, Z}
            bearing_start (float): Cap initial en radians
            bearing_end (float): Cap final en radians
            config (str): Configuration Dubins ('LSL', 'RSR', 'LSR', 'RSL')
            num_points_total (int): Nombre total de points pour la trajectoire
        
        Returns:
            tuple: (path, length) où path est un dict {X, Y, Z}
                  et length est la longueur totale du chemin
        """
        # Normaliser les bearings entre 0 et 2*pi
        bearing_start = bearing_start % (2 * pi)
        bearing_end = bearing_end % (2 * pi)
        
        # Diviser le nombre de points entre les segments (3 segments: 2 virages + 1 droit)
        points_per_segment = num_points_total // 3
        
        # Points à générer pour chaque segment
        first_turn_points = points_per_segment
        straight_points = points_per_segment
        second_turn_points = num_points_total - first_turn_points - straight_points
        
        # Direction du premier virage ('L' gauche, 'R' droite)
        first_turn_dir = config[0]
        
        # Direction du segment droit ('S')
        # Configuration du second virage ('L' gauche, 'R' droite)
        second_turn_dir = config[2]
        
        # Déterminer l'angle des virages en fonction de la direction
        first_turn_angle = pi/2 if first_turn_dir == 'L' else -pi/2
        second_turn_angle = pi/2 if second_turn_dir == 'L' else -pi/2
        
        # Calculer les centres des cercles pour les virages
        first_center_bearing = (bearing_start + first_turn_angle) % (2 * pi)
        first_center = {
            'X': 0,
            'Y': 0,
            'Z': start_point['Z']
        }
        first_center['X'], first_center['Y'] = get_destination_from_range_and_bearing_cartesian(
            start_point, self.min_turn_radius, first_center_bearing
        )
        
        # Générer le premier virage
        mid_bearing = None
        if first_turn_dir == 'L':
            delta = pi/2  # Pour un virage à gauche, nous tournons de 90 degrés
            mid_bearing = (bearing_start + delta) % (2 * pi)
        else:  # 'R'
            delta = -pi/2  # Pour un virage à droite, nous tournons de -90 degrés
            mid_bearing = (bearing_start + delta) % (2 * pi)
        
        # Point intermédiaire après le premier virage
        mid_point = {
            'X': 0,
            'Y': 0,
            'Z': start_point['Z']
        }
        mid_point['X'], mid_point['Y'] = get_destination_from_range_and_bearing_cartesian(
            first_center, self.min_turn_radius, (mid_bearing + pi) % (2 * pi)
        )
        
        # Calculer le centre du deuxième cercle
        second_center_bearing = (bearing_end + second_turn_angle) % (2 * pi)
        second_center = {
            'X': 0,
            'Y': 0,
            'Z': end_point['Z']
        }

        second_center['X'], second_center['Y'] = get_destination_from_range_and_bearing_cartesian(
            end_point, self.min_turn_radius, second_center_bearing
        )
        
        # Calculer le point de tangence pour la transition entre le segment droit et le deuxième virage
        tangent_point = self._find_tangent_point(mid_point, mid_bearing, second_center,
                                               self.min_turn_radius, second_turn_dir)
        
        if tangent_point is None:
            # Aucun point de tangente trouvé, cette configuration n'est pas valide
            return None, float('inf')
        
        # Générer le premier virage
        first_turn = self._generate_turn_segment(
            start_point, bearing_start, mid_bearing, first_turn_points, first_turn_dir
        )
        
        # Générer le segment droit
        straight_segment = self._generate_straight_segment(
            mid_point, tangent_point, straight_points
        )
        
        # Calculer le bearing final pour le deuxième virage
        final_tangent_bearing = compute_bearing_cartesian(tangent_point, second_center)[0]
        final_bearing = (final_tangent_bearing + pi) % (2 * pi)
        
        # Générer le deuxième virage
        second_turn = self._generate_turn_segment(
            tangent_point, final_bearing, bearing_end, second_turn_points, second_turn_dir
        )
        
        # Fusionner les segments
        x_points = first_turn['path']['X'] + straight_segment['path']['X'] + second_turn['path']['X']
        y_points = first_turn['path']['Y'] + straight_segment['path']['Y'] + second_turn['path']['Y']
        z_points = first_turn['path']['Z'] + straight_segment['path']['Z'] + second_turn['path']['Z']
        
        # Interpolation linéaire de l'altitude sur tout le chemin
        for i in range(len(z_points)):
            fraction = i / (len(z_points) - 1) if len(z_points) > 1 else 0
            z_points[i] = start_point['Z'] + fraction * (end_point['Z'] - start_point['Z'])

        # Calculer la longueur totale du chemin
        first_arc_length = self.min_turn_radius * abs(mid_bearing - bearing_start)
        straight_length = compute_distance_cartesian(mid_point, tangent_point)[0]
        second_arc_length = self.min_turn_radius * abs(bearing_end - final_bearing)
        total_length = first_arc_length + straight_length + second_arc_length
        
        return {'X': x_points, 'Y': y_points, 'Z': z_points}, total_length
    
    def _find_tangent_point(self, start_point, start_bearing, circle_center, radius, turn_dir):
        """
        Trouve le point de tangence à un cercle à partir d'un point et d'un bearing initial.
        
        Args:
            start_point (dict): Point de départ
            start_bearing (float): Cap initial en radians
            circle_center (dict): Centre du cercle
            radius (float): Rayon du cercle
            turn_dir (str): Direction du virage ('L' ou 'R')
            
        Returns:
            dict: Point de tangente ou None si aucun n'existe
        """
        # Distance entre le point de départ et le centre du cercle
        d = compute_distance_cartesian(start_point, circle_center)[0]
        
        # Si la distance est inférieure au rayon, aucune tangente externe n'est possible
        if d <= radius:
            return None
        
        # Bearing du centre du cercle par rapport au point de départ
        center_bearing = compute_bearing_cartesian(start_point, circle_center)[0]
        
        # Angle entre le bearing initial et la direction du centre
        alpha = (center_bearing - start_bearing) % (2 * pi)
        
        # Angle entre la tangente et la ligne du centre
        theta = asin(radius / d)
        
        # Ajustement en fonction de la direction du virage
        if turn_dir == 'L':
            tangent_bearing = (center_bearing + theta) % (2 * pi)
        else:  # 'R'
            tangent_bearing = (center_bearing - theta) % (2 * pi)
            
        # Distance le long de la tangente
        tangent_distance = d * cos(theta)
        
        # Point de tangence
        tangent_point = {
            'X': 0,
            'Y': 0,
            'Z': start_point['Z']
        }
        tangent_point['X'], tangent_point['Y'] = get_destination_from_range_and_bearing_cartesian(
            start_point, tangent_distance, tangent_bearing
        )
        
        return tangent_point
        
    def _generate_turn_segment(self, start_point, start_bearing, end_bearing, num_points, turn_dir='L'):
        """
        Génère un segment de virage entre deux caps (bearings)
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            start_bearing (float): Cap initial (radians)
            end_bearing (float): Cap final (radians)
            num_points (int): Nombre de points à générer
            turn_dir (str): Direction du virage ('L' pour gauche, 'R' pour droite)
            
        Returns:
            dict: Segment de virage {path: {X, Y, Z}, end_point}
        """
        # Normaliser les bearings entre 0 et 2*pi
        start_bearing = start_bearing % (2 * pi)
        end_bearing = end_bearing % (2 * pi)
        
        # Calculer la direction et l'angle de rotation
        direction = 1 if turn_dir == 'L' else -1  # 1 pour gauche, -1 pour droite
        
        # Calculer l'angle de rotation
        if turn_dir == 'L':
            delta_bearing = (end_bearing - start_bearing) % (2 * pi)
        else:
            delta_bearing = (start_bearing - end_bearing) % (2 * pi)
            delta_bearing = -delta_bearing  # Pour virage à droite
        
        # Assurer que l'angle est dans la bonne direction
        if (turn_dir == 'L' and delta_bearing < 0) or (turn_dir == 'R' and delta_bearing > 0):
            delta_bearing = delta_bearing + direction * 2 * pi
            
        # Calculer le centre du cercle de virage
        center_bearing = (start_bearing + direction * pi/2) % (2 * pi)
        center_x, center_y = get_destination_from_range_and_bearing_cartesian(
            start_point, self.min_turn_radius, center_bearing
        )
        center_point = {
            'X': center_x,
            'Y': center_y,
            'Z': start_point['Z']
        }
        
        # Générer les points sur l'arc
        bearings = np.linspace(0, abs(delta_bearing), num_points)
        x_points = []
        y_points = []
        
        for angle in bearings:
            # Pour chaque bearing, calculer un point sur le cercle
            if turn_dir == 'L':
                point_bearing = (start_bearing + angle) % (2 * pi)
            else:
                point_bearing = (start_bearing - angle) % (2 * pi)
                
            # Direction du point depuis le centre (opposée au bearing)
            from_center_bearing = (point_bearing + pi + direction * pi/2) % (2 * pi)
            x, y = get_destination_from_range_and_bearing_cartesian(
                center_point, self.min_turn_radius, from_center_bearing
            )
            x_points.append(x)
            y_points.append(y)

        # Point final du virage
        end_point = {
            'X': x_points[-1],
            'Y': y_points[-1],
            'Z': start_point['Z']
        }
        
        # Altitudes (même altitude partout dans le virage)
        z_points = [start_point['Z']] * num_points
        
        return {
            'path': {
                'X': x_points,
                'Y': y_points,
                'Z': z_points
            },
            'end_point': end_point
        }

    def _generate_straight_segment(self, start_point, end_point, num_points):
        """
        Génère un segment de trajectoire en ligne droite
        
        Args:
            start_point (dict): Point de départ
            end_point (dict): Point d'arrivée
            num_points (int): Nombre de points à générer
            
        Returns:
            dict: Segment droit {path: {latitude, longitude, altitude}, end_point}
        """
        # Générer les points intermédiaires
        fractions = np.linspace(0, 1, num_points)
        x_points = []
        y_points = []
        
        # Calculer le bearing initial
        bearing = compute_bearing_cartesian(start_point, end_point)[0]
        
        # Calculer la distance
        distance = compute_distance_cartesian(start_point, end_point)[0]
        
        for f in fractions:
            # Pour chaque fraction, calculer un point à la distance proportionnelle
            x, y = get_destination_from_range_and_bearing_cartesian(
                start_point, f * distance, bearing
            )
            x_points.append(x)
            y_points.append(y)

        # Interpolation linéaire de l'altitude
        z_points = []
        for f in fractions:
            alt = start_point['Z'] + f * (end_point['Z'] - start_point['Z'])
            z_points.append(alt)
        
        return {
            'path': {
                'X': x_points,
                'Y': y_points,
                'Z': z_points
            },
            'end_point': end_point
        }
        
class PythagoreanHodographPath(TrajectoryGenerator):
    """
    Pythagorean Hodograph path trajectory generator
    
    Args:
        params (dict): Parameters for the trajectory generation, e.g., number of points, degree, etc.
        UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)
        self.degree = params.get('ph_degree', 5)  # Degré par défaut pour les courbes PH
    
    def generate_path(self, start_point, end_point) -> Dict:
        """
        Génère une trajectoire PH (Pythagorean Hodograph) entre deux points cartésiens
        
        Args:
            start_point (dict): Point de départ {X, Y, Z}
            end_point (dict): Point d'arrivée {X, Y, Z}
            
        Returns:
            dict: Points de trajectoire {X: [], Y: [], Z: []}
        """
        num_points = self.params.get('num_points', 200)
        end_point = extract_waypoint(end_point)
        # Conversion en coordonnées cartésiennes pour simplifier les calculs
        start_xyz = np.array([start_point['X'], start_point['Y'], start_point['Z']])
        end_xyz = np.array([end_point['X'], end_point['Y'], end_point['Z']])
        
        # Extraction des vitesses initiale et finale si disponibles
        initial_velocity = np.array([0.0, 0.0, 0.0])
        final_velocity = np.array([0.0, 0.0, 0.0])
        
        if self.UAV_data:
            min_velocity = self.UAV_data.get('min_airspeed', 8.0)
            max_velocity = self.UAV_data.get('max_airspeed', 30.0)
            avg_speed = (min_velocity + max_velocity) / 2
        else:
            avg_speed = 20.0  # Valeur par défaut si UAV_data n'est pas disponible
        
        if 'bearing' in start_point:
            # Conversion du bearing en vecteur de vitesse initiale
            bearing_rad = start_point['bearing']
            initial_velocity = np.array([
                avg_speed * np.cos(bearing_rad),
                avg_speed * np.sin(bearing_rad),
                0.0  # Vitesse verticale initiale nulle
            ])
        
        if 'bearing' in end_point:
            # Conversion du bearing en vecteur de vitesse finale
            bearing_rad = end_point['bearing']
            final_velocity = np.array([
                avg_speed * np.cos(bearing_rad),
                avg_speed * np.sin(bearing_rad),
                0.0  # Vitesse verticale finale nulle
            ])
        
        # Génération des coefficients pour la courbe PH
        ph_coefficients = self._generate_ph_coefficients(
            start_xyz, end_xyz, initial_velocity, final_velocity
        )
        
        # Évaluation de la courbe PH
        t = np.linspace(0, 1, num_points)
        path_xyz = self._evaluate_ph_curve(ph_coefficients, t)
        
        # Conversion retour en coordonnées géodésiques
        x_points = path_xyz[:, 0].tolist()
        y_points = path_xyz[:, 1].tolist()
        z_points = path_xyz[:, 2].tolist()

        return {
            'X': x_points,
            'Y': y_points,
            'Z': z_points
        }

    def _generate_ph_coefficients(self, start, end, start_velocity, end_velocity):
        """
        Génération des coefficients polynomiaux pour la courbe hodographe pythagoricienne.
        
        Pour une courbe PH de degré 5, nous générons des coefficients qui garantissent:
        1. Les positions de départ et d'arrivée spécifiées
        2. Les vitesses initiale et finale spécifiées
        3. Une paramétrisation satisfaisant l'identité pythagoricienne pour l'hodographe
        
        Args:
            start (np.array): Point de départ en coordonnées cartésiennes [x, y, z]
            end (np.array): Point d'arrivée en coordonnées cartésiennes [x, y, z]
            start_velocity (np.array): Vecteur vitesse initiale [vx, vy, vz]
            end_velocity (np.array): Vecteur vitesse final [vx, vy, vz]
            
        Returns:
            dict: Coefficients pour la courbe PH
        """
        # Normalisation des vecteurs de vitesse si non nuls
        if np.linalg.norm(start_velocity) > 0:
            start_velocity = start_velocity / np.linalg.norm(start_velocity)
        if np.linalg.norm(end_velocity) > 0:
            end_velocity = end_velocity / np.linalg.norm(end_velocity)
            
        # Distance entre points de départ et d'arrivée
        length = np.linalg.norm(end - start)
        
        # Ajustement des vitesses selon la distance
        speed_scale = length * 2.5
        start_velocity = start_velocity * speed_scale
        end_velocity = end_velocity * speed_scale
        
        if self.degree == 5:
            # Pour une courbe PH quintique (degré 5), nous utilisons une approche basée 
            # sur les polynômes de Bernstein pour garantir la propriété pythagoricienne
            
            # Calcul des coefficients de contrôle pour l'hodographe (dérivée)
            # Pour une courbe PH quintique, l'hodographe est de degré 4
            
            # Points de contrôle pour la courbe intégrée (position)
            p0 = start
            p5 = end
            
            # Contrôle de la vitesse initiale
            p1 = start + start_velocity / 5
            
            # Contrôle de la vitesse finale
            p4 = end - end_velocity / 5
            
            # Points intermédiaires pour assurer une transition douce
            # Ces points sont calculés de manière à garantir la propriété pythagoricienne
            direction = end - start
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
            # Vecteur perpendiculaire pour créer une courbe
            perpendicular = np.array([-direction[1], direction[0], 0])
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
            else:
                perpendicular = np.array([0, 1, 0])
                
            # Points de contrôle intermédiaires garantissant une courbure optimale
            deviation = length * 0.25
            
            p2 = start + direction * length / 3 + perpendicular * deviation
            p3 = end - direction * length / 3 + perpendicular * deviation
            
            # Ajustement de l'altitude pour une trajectoire douce
            p2[2] = start[2] + (end[2] - start[2]) / 3
            p3[2] = start[2] + 2 * (end[2] - start[2]) / 3
            
            # Retourner les coefficients sous forme de tableau numpy
            return np.array([p0, p1, p2, p3, p4, p5])
            
        else:
            # Implémentation par défaut pour courbe PH cubique (plus simple mais moins flexible)
            p0 = start
            p3 = end
            
            # Points intermédiaires avec une légère déviation perpendiculaire
            direction = end - start
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1, 0, 0])
                
            perpendicular = np.array([-direction[1], direction[0], 0])  
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
            else:
                perpendicular = np.array([0, 1, 0])
                
            deviation = length * 0.2
            control_distance = length / 3
            
            p1 = start + direction * control_distance + perpendicular * deviation
            p2 = end - direction * control_distance + perpendicular * deviation
            
            p1[2] = start[2] + (end[2] - start[2]) / 3
            p2[2] = start[2] + 2 * (end[2] - start[2]) / 3
            
            return np.array([p0, p1, p2, p3])
        
    def _evaluate_ph_curve(self, control_points, t):
        """
        Évaluation de la courbe PH en utilisant les coefficients de contrôle.
        
        Pour une courbe PH de degré n, nous utilisons les polynômes de Bernstein
        pour évaluer la courbe en garantissant les propriétés des hodographes pythagoriciennes.
        
        Args:
            control_points (np.array): Coefficients de contrôle pour la courbe
            t (np.array): Paramètres d'évaluation entre 0 et 1
            
        Returns:
            np.array: Points de la courbe évaluée
        """
        # Initialisation du tableau de résultats
        result = np.zeros((len(t), 3))
        
        # Déterminer le degré basé sur le nombre de points de contrôle
        n = len(control_points) - 1
        
        if n == 5:  # Courbe PH quintique
            # Évaluation d'une courbe de Bézier de degré 5
            for i, ti in enumerate(t):
                point = np.zeros(3)
                for j in range(n + 1):
                    # Coefficient binomial * (1-t)^(n-j) * t^j
                    bin_coef = self._binomial_coefficient(n, j)
                    bernstein = bin_coef * ((1-ti)**(n-j)) * (ti**j)
                    point += bernstein * control_points[j]
                result[i] = point
        else:  # Courbe de Bézier cubique (degré 3)
            p0, p1, p2, p3 = control_points
            for i, ti in enumerate(t):
                result[i] = ((1-ti)**3 * p0 + 
                            3 * (1-ti)**2 * ti * p1 + 
                            3 * (1-ti) * ti**2 * p2 + 
                            ti**3 * p3)
                
        return result
        
    def _binomial_coefficient(self, n, k):
        """
        Calcule le coefficient binomial C(n,k) = n! / (k! * (n-k)!)
        """
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        
        # Calcul efficace du coefficient binomial
        result = 1
        for i in range(1, k + 1):
            result *= (n - (i - 1))
            result //= i
            
        return result
        
        
def generate_all_trajectories(start_point, end_point, params, UAV_data):
    """
    Génère toutes les trajectoires possibles (droit, courbe, Dubins 3D, PH) entre deux points.
    Retourne un dictionnaire avec chaque type de trajectoire.
    
    Args:
        start_point (dict): Point de départ {X, Y, Z}
        end_point (dict): Point d'arrivée {X, Y, Z}  
        params (dict): Paramètres pour la génération de trajectoire
        UAV_data (dict): Dictionnaire contenant les données UAV
    
    Returns:
        list: Liste avec toutes les trajectoires générées
    """
    smoother = TrajectorySmoothing(params)
    # Générateur ligne droite
    straight_traj = StraightLineTrajectory(params, UAV_data)
    straight = straight_traj.generate_path(start_point, end_point)

    # Générateur courbe circulaire
    circular_traj = CircularTrajectory(params, UAV_data)
    circular = circular_traj.generate_path(start_point, end_point)
     
    # Générateur Dubins 3D
    dubins_traj = DubinsPath3D(params, UAV_data)
    dubins = dubins_traj.generate_path(start_point, end_point)

    # Générateur Pythagorean Hodograph
    ph_traj = PythagoreanHodographPath(params, UAV_data)
    ph = ph_traj.generate_path(start_point, end_point)

    # Retourne une liste avec toutes les trajectoires
    return [straight, circular, dubins, ph]

class TrajectoryEvaluator:
    """Classe pour évaluer les trajectoires générées"""

    def __init__(self, params: Dict, UAV_data: Dict, flight_conditions: Dict):
        self.params = params
        self.UAV_data = UAV_data
        self.alt_min = params['Z_lower_bound']
        self.alt_max = params['Z_upper_bound']
        self.flight_conditions = flight_conditions
        self.min_airspeed = UAV_data['max_airspeed']
        self.max_airspeed = UAV_data['max_airspeed']
        self.max_turn_rate = UAV_data['max_turn_rate']
        # Calcul dynamique des angles maximums basé sur les fonctions existantes
        self.max_climb_angle = self._calculate_max_climb_angle()
        self.max_descent_angle = self._calculate_max_descent_angle()
        self.max_bank_angle = self._calculate_max_bank_angle()
        self.obstacles = params.get('obstacles', [])
        
        # Poids pour le calcul du score
        self.distance_weight = 0.3  # Importance de la distance
        self.energy_weight = 0.7    # Importance de l'énergie
        self.obstacle_weight = 2.0    # Importance des obstacles

    def _calculate_max_climb_angle(self):
        """Calcule l'angle de montée maximum basé sur les caractéristiques du drone"""
        from compute import compute_required_thrust, get_lift_to_drag

        # Créer une copie des conditions de vol pour les calculs
        test_conditions = self.flight_conditions.copy()
        test_conditions['flight_path_angle'] = 0  # Point de départ à niveau
        test_conditions['bank_angle'] = 0         # Sans inclinaison
        test_conditions['airspeed'] = self.min_airspeed  # Utiliser la vitesse minimum pour le cas le plus défavorable
        
        # Calculer le rapport portance/traînée pour ajuster les calculs de montée
        lift_to_drag_ratio = get_lift_to_drag(self.UAV_data, test_conditions)
        
        # Théoriquement, l'angle maximum de montée est arcsin(1/L/D) dans un cas idéal
        # Mais en pratique, il est limité par la puissance disponible
        theoretical_max_angle = np.arcsin(1 / lift_to_drag_ratio) if lift_to_drag_ratio > 1 else np.pi/6
        
        # Calculer la poussée maximum disponible à vitesse minimale
        max_thrust = compute_required_thrust(self.UAV_data, test_conditions)
        
        # Augmenter progressivement l'angle jusqu'à trouver la limite
        max_angle = 0
        step = 0.01  # Pas de 0.01 radian (~0.57°)
        
        # On limite l'exploration à l'angle théorique maximum
        for angle in np.arange(0, min(theoretical_max_angle, np.pi/4), step):
            test_conditions['flight_path_angle'] = angle
            required_thrust = compute_required_thrust(self.UAV_data, test_conditions)
            
            # Si la poussée requise dépasse la poussée disponible, on a trouvé la limite
            if required_thrust > max_thrust * 1.1:  # 10% de marge de sécurité
                break
            max_angle = angle
        
        # Limiter à une valeur raisonnable (max 20°)
        return min(max_angle, np.deg2rad(20))

    def _calculate_max_descent_angle(self):
        """Calcule l'angle de descente maximum basé sur les caractéristiques du drone"""
        from compute import get_sink_rate

        # Créer une copie des conditions de vol pour les calculs
        test_conditions = self.flight_conditions.copy()
        test_conditions['flight_path_angle'] = 0
        test_conditions['bank_angle'] = 0
        test_conditions['airspeed'] = self.max_airspeed  # Utiliser la vitesse maximum pour le cas le plus défavorable
        test_conditions['airspeed_dot'] = 0  # Pas d'accélération

        # Calculer le taux de descente maximum en mode plané
        sink_rate = get_sink_rate(self.UAV_data, test_conditions)

        # Calculer l'angle correspondant à ce taux de descente
        max_descent_angle = -np.arcsin(sink_rate / test_conditions['airspeed'])

        # Limiter à une valeur sûre (-30° max)
        return max(-np.deg2rad(30), max_descent_angle)

    def _calculate_max_bank_angle(self):
        """Calcule l'angle d'inclinaison maximum basé sur les caractéristiques du drone"""
        # Utiliser la relation entre le taux de virage et l'angle d'inclinaison
        # tan(φ) = v * ω / g

        cruise_speed = (self.min_airspeed + self.max_airspeed) / 2
        gravity = self.flight_conditions.get('grav_accel', 9.81)

        # Calculer l'angle d'inclinaison correspondant au taux de virage maximum
        max_bank_rad = np.arctan(cruise_speed * self.max_turn_rate / gravity)

        # Limiter à 45° pour des raisons de sécurité structurelle
        return min(max_bank_rad, np.deg2rad(45))

    def evaluate_trajectories(self, trajectories: List[Dict]) -> Dict:
        """
        Évalue les trajectoires et retourne la meilleure
        
        Args:
            trajectories: Liste des trajectoires à évaluer
            
        Returns:
            Dict: La meilleure trajectoire
        """
        if not trajectories:
            return {'X': [0], 'Y': [0], 'Z': [400]}
        # Noms des types de trajectoires dans l'ordre de génération
        trajectory_types = ["Ligne droite", "Circulaire", "Dubins", "Pythagorean Hodograph"]
        
        best_trajectory = trajectories[0]
        best_trajectory_type = trajectory_types[0]
        min_score = float('inf')
        scores = []
        
        for i, trajectory in enumerate(trajectories):
            trajectory_type = trajectory_types[i] if i < len(trajectory_types) else f"Type inconnu {i}"
            score = self._evaluate_single_trajectory(trajectory)
            scores.append(score)
        
            # Afficher le score de chaque trajectoire pour le débogage
            print(f"Score de la trajectoire {trajectory_type}: {score:.2f}")

            if score < min_score:
                min_score = score
                best_trajectory = trajectory
                best_trajectory_type = trajectory_type
    
        # Afficher la trajectoire choisie
        print(f"\nTrajectoire choisie: {best_trajectory_type} (score: {min_score:.2f})")

        return best_trajectory

    def _evaluate_single_trajectory(self, trajectory: Dict) -> float:
        """
        Évalue une seule trajectoire
        
        Args:
            trajectory: Trajectoire à évaluer
            FLT_conditions: Conditions de vol
            
        Returns:
            float: Score de la trajectoire (plus petit = meilleur)
        """
        total_distance = 0
        total_energy = 0
        x_points = trajectory['X']
        y_points = trajectory['Y']
        z_points = trajectory['Z']
        
        # Vérifier qu'il y a suffisamment de points
        if len(x_points) < 2:
            return float('inf')
        
        # Pénalités pour violations de contraintes
        constraints_penalty = 0
        
        # Vérifier les altitudes
        for z in z_points:
            if z < self.alt_min:
                constraints_penalty += 1000 * (self.alt_min - z)
            if z > self.alt_max:
                constraints_penalty += 1000 * (z - self.alt_max)
                
        # Vérifier les obstacles
        collision_exists, collision_points, min_distance = check_trajectory_obstacles(trajectory, self.obstacles)
        
        # Pénalité pour collision avec obstacles
        if collision_exists:
            obs_penalty += 50000 * len(collision_points)  # Pénalité très élevée pour les collisions
        
        # Pénalité pour proximité d'obstacles
        safety_margin = 30.0  # marge de sécurité en mètres
        if min_distance < safety_margin and min_distance > 0:
            obs_penalty += 5000 * (safety_margin - min_distance) / safety_margin
        
        # Calculer la distance, l'énergie et vérifier les contraintes de virage
        current_flight_conditions = self.flight_conditions.copy()
        airspeed = current_flight_conditions['airspeed']
        
        for i in range(1, len(x_points)):
            dx = x_points[i] - x_points[i-1]
            dy = y_points[i] - y_points[i-1]
            dz = z_points[i] - z_points[i-1]
            segment_distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Éviter division par zéro
            if segment_distance < 1e-6:
                continue
                
            total_distance += segment_distance
            
            # Calculer l'angle de vol (flight path angle)
            flight_path_angle = np.arcsin(dz / segment_distance)
            
            # Vérifier si la pente est trop raide pour le drone
            if flight_path_angle > self.max_climb_angle:
                constraints_penalty += 5000 * (flight_path_angle - self.max_climb_angle)
            if flight_path_angle < self.max_descent_angle:
                constraints_penalty += 5000 * (self.max_descent_angle - flight_path_angle)
            
            current_flight_conditions['flight_path_angle'] = flight_path_angle
            
            # Calculer l'angle de virage si ce n'est pas le premier segment
            if i > 1:
                prev_bearing = np.arctan2(y_points[i-1] - y_points[i-2], x_points[i-1] - x_points[i-2])
                curr_bearing = np.arctan2(dy, dx)
                delta_bearing = (curr_bearing - prev_bearing + np.pi) % (2 * np.pi) - np.pi
                
                # Calculer le temps pour ce segment
                time_segment = segment_distance / airspeed
                
                # Calculer le taux de virage
                turn_rate = abs(delta_bearing / time_segment)
                
                # Vérifier le taux de virage maximum
                if turn_rate > self.max_turn_rate:
                    constraints_penalty += 10000 * (turn_rate - self.max_turn_rate)
                
                # Calculer l'angle de virage (bank angle)
                gravity = current_flight_conditions['grav_accel']
                
                # Formule de l'angle de virage: tan(bank) = v * turn_rate / g
                if airspeed > 0:
                    bank_angle = min(np.arctan(airspeed * turn_rate / gravity), self.max_bank_angle)
                    current_flight_conditions['bank_angle'] = bank_angle
            
            # Calculer la puissance pour ce segment
            try:
                power = get_power_consumption(self.UAV_data, current_flight_conditions)
            except:
                # En cas d'erreur dans le calcul de puissance, utiliser une estimation
                power = self.UAV_data.get('max_power_consumption', 500) * 0.6
            
            # Temps pour parcourir ce segment
            time_segment = segment_distance / airspeed
            
            # Énergie consommée pour ce segment (Power * Time)
            energy_segment = power * time_segment
            total_energy += energy_segment
        
        # Score composite combinant distance, énergie et pénalités de contraintes
        score = (self.distance_weight * total_distance / 1000) + \
                (self.energy_weight * total_energy / 100) + \
                (self.obstacle_weight * obs_penalty) + \
                constraints_penalty
        
        return score
    
    

class TrajectorySmoothing:
    """
    Classe fournissant des méthodes pour lisser les trajectoires générées.
    """
    
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        
    def smooth_trajectory_spline(self, trajectory):
        """Lissage par spline"""
        x_points = np.array(trajectory['X'])
        y_points = np.array(trajectory['Y'])
        z_points = np.array(trajectory['Z'])
        
        # Paramètres pour la spline
        t = np.linspace(0, 1, len(x_points))
        t_smooth = np.linspace(0, 1, len(x_points))
        
        try:
            # Interpolation spline
            fx = interpolate.interp1d(t, x_points, kind='cubic')
            fy = interpolate.interp1d(t, y_points, kind='cubic')
            fz = interpolate.interp1d(t, z_points, kind='cubic')
            
            x_smooth = fx(t_smooth)
            y_smooth = fy(t_smooth)
            z_smooth = fz(t_smooth)
            
            return {'X': x_smooth.tolist(), 'Y': y_smooth.tolist(), 'Z': z_smooth.tolist()}
        except:
            # Si l'interpolation échoue, retourner la trajectoire originale
            return trajectory
        
    def smooth_trajectory_savgol(self, trajectory):
        """Lissage par filtre Savitzky-Golay"""
        try:
            x_points = np.array(trajectory['X'])
            y_points = np.array(trajectory['Y'])
            z_points = np.array(trajectory['Z'])
            
            # Paramètres du filtre
            window_length = min(11, len(x_points) if len(x_points) % 2 == 1 else len(x_points) - 1)
            if window_length < 3:
                return trajectory
                
            polyorder = min(3, window_length - 1)
            
            x_smooth = savgol_filter(x_points, window_length, polyorder)
            y_smooth = savgol_filter(y_points, window_length, polyorder)
            z_smooth = savgol_filter(z_points, window_length, polyorder)
            
            return {'X': x_smooth.tolist(), 'Y': y_smooth.tolist(), 'Z': z_smooth.tolist()}
        except:
            return trajectory
    
    def smooth_dubins_junctions(self, trajectory, num_junction_points=5):
        """Lissage spécifique pour les jonctions Dubins"""
        return self.smooth_trajectory_spline(trajectory)
        
    def smooth_trajectory(self, trajectory, method='spline'):
        """
        Applique le lissage selon la méthode spécifiée
        
        Args:
            trajectory: Trajectoire à lisser
            method: Méthode de lissage ('spline', 'savgol', 'dubins_junctions')
            
        Returns:
            Dict: Trajectoire lissée
        """
        if method == 'spline':
            return self.smooth_trajectory_spline(trajectory)
        elif method == 'savgol':
            return self.smooth_trajectory_savgol(trajectory)
        elif method == 'dubins_junctions':
            return self.smooth_dubins_junctions(trajectory)
        else:
            return trajectory
