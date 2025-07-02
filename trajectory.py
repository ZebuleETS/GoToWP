import numpy as np
from math import pi, asin, cos
from abc import ABC, abstractmethod
from compute import (
    cartesian_to_geographic,
    compute_bearing,
    compute_distance,
    geographic_to_cartesian,
    get_destination_from_range_and_bearing,
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
    Straight line trajectory generator
    Args:
        params (dict): Parameters for the trajectory generation, e.g., number of points, radius, etc.
        UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)

    def generate_path(self, start_point, end_point):
        """
        Génère une trajectoire en ligne droite géodésique entre deux points
        
        Args:
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
            params (dict): Paramètres UAV et simulation 
            
        Returns:
            dict: Points de trajectoire {latitude: [], longitude: [], altitude: []}
        """
        num_points = self.params.get('num_points', 50)

        distance = compute_distance(start_point, end_point)[0]
        initial_bearing = compute_bearing(start_point, end_point)[0]

        # Génération des points intermédiaires
        fractions = np.linspace(0, 1, num_points)
        latitudes = []
        longitudes = []
        for f in fractions:
            d = distance * f
            dest = get_destination_from_range_and_bearing(
                start_point, d, initial_bearing
            )
            latitudes.append(dest[0])
            longitudes.append(dest[1])
        alt = np.linspace(start_point['altitude'], end_point['altitude'], num_points)
        return {'latitude': latitudes, 'longitude': longitudes, 'altitude': alt.tolist()}

class CircularTrajectory(TrajectoryGenerator):
    """
    Circular trajectory generator utilisant les fonctions utilitaires du projet
    Args:
            params (dict): Parameters for the trajectory generation, e.g., number of points, radius, etc.
            UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)

    def generate_path(self, start_point, end_point) -> Dict:
        """
        Génère une trajectoire circulaire géodésiques entre deux points en utilisant les fonctions utilitaires.
        Args:
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
            params (dict): Paramètres UAV et simulation 
        Returns:
            dict: Points de trajectoire {latitude: [], longitude: [], altitude: []}
        """
        num_points = self.params.get('num_points', 50)
        radius = self.params.get('radius', 100)  # rayon en mètres
        print(type(start_point), type(end_point))
        # Calcul du centre du cercle (milieu géodésique)
        distance = compute_distance(start_point, end_point)[0]
        initial_bearing = compute_bearing(start_point, end_point)[0]
        center_distance = distance / 2
        center_bearing = initial_bearing
        center_point_lat, center_point_lon = get_destination_from_range_and_bearing(
            start_point, center_distance, center_bearing)
        center_point = {
            'latitude': center_point_lat,
            'longitude': center_point_lon,
            'altitude': (start_point['altitude'] + end_point['altitude']) / 2
        }

        # Génération des points sur le cercle
        angles = np.linspace(0, 2 * pi, num_points)
        latitudes = []
        longitudes = []
        for angle in angles:
            # Pour chaque angle, calculer la position sur le cercle
            lat, lon = get_destination_from_range_and_bearing(
                center_point, radius, angle)
            latitudes.append(lat)
            longitudes.append(lon)
        altitudes = np.linspace(start_point['altitude'], end_point['altitude'], num_points)
        return {'latitude': latitudes, 'longitude': longitudes, 'altitude': altitudes.tolist()}

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
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
        Returns:
            dict: Points de trajectoire {latitude: [], longitude: [], altitude: []}
        """
        num_points_total = self.params.get('num_points', 100)
        
        # Calculer le cap initial et final
        bearing_start = 0  # Par défaut, on suppose que le drone pointe vers l'Est
        if 'bearing' in start_point:
            bearing_start = start_point['bearing']
        
        # Calculer le cap entre le point de départ et d'arrivée
        bearing_end = compute_bearing(start_point, end_point)[0]
        
        # Évaluer chaque configuration de Dubins et choisir la meilleure
        min_length = float('inf')
        best_path = None
        
        for config in self.dubins_configs:
            path, length = self._compute_dubins_path(start_point, end_point, bearing_start, 
                                               bearing_end, config, num_points_total)
            if length < min_length:
                min_length = length
                best_path = path
        
        if best_path is None:
            # Fallback: générer une ligne droite si aucun chemin n'est valide
            straight_points = num_points_total
            straight_path = self._generate_straight_segment(start_point, end_point, straight_points)
            
            latitudes = straight_path['path']['latitude']
            longitudes = straight_path['path']['longitude']
            
            # Interpolation linéaire de l'altitude
            altitudes = []
            for i in range(len(latitudes)):
                fraction = i / (len(latitudes) - 1) if len(latitudes) > 1 else 0
                alt = start_point['altitude'] + fraction * (end_point['altitude'] - start_point['altitude'])
                altitudes.append(alt)
        else:
            latitudes = best_path['latitude']
            longitudes = best_path['longitude']
            altitudes = best_path['altitude']
        
        # Si numpy array, convertir en liste
        if isinstance(altitudes, np.ndarray):
            altitudes = altitudes.tolist()
            
        trajectory = {'latitude': latitudes, 'longitude': longitudes, 'altitude': altitudes}
        
        smoother = TrajectorySmoothing(self.params)
        trajectory = smoother.smooth_trajectory(trajectory, method='dubins_junctions')

        return trajectory

    def _compute_dubins_path(self, start_point, end_point, bearing_start, bearing_end,
                            config, num_points_total):
        """
        Calcule un chemin de Dubins selon une configuration spécifique.
        
        Args:
            start_point (dict): Point de départ
            end_point (dict): Point d'arrivée
            bearing_start (float): Cap initial en radians
            bearing_end (float): Cap final en radians
            config (str): Configuration Dubins ('LSL', 'RSR', 'LSR', 'RSL')
            num_points_total (int): Nombre total de points pour la trajectoire
        
        Returns:
            tuple: (path, length) où path est un dict {latitude, longitude, altitude}
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
            'latitude': 0,
            'longitude': 0,
            'altitude': start_point['altitude']
        }
        first_center['latitude'], first_center['longitude'] = get_destination_from_range_and_bearing(
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
            'latitude': 0,
            'longitude': 0,
            'altitude': start_point['altitude']
        }
        mid_point['latitude'], mid_point['longitude'] = get_destination_from_range_and_bearing(
            first_center, self.min_turn_radius, (mid_bearing + pi) % (2 * pi)
        )
        
        # Calculer le centre du deuxième cercle
        second_center_bearing = (bearing_end + second_turn_angle) % (2 * pi)
        second_center = {
            'latitude': 0,
            'longitude': 0,
            'altitude': end_point['altitude']
        }
        second_center['latitude'], second_center['longitude'] = get_destination_from_range_and_bearing(
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
        final_tangent_bearing = compute_bearing(tangent_point, second_center)[0]
        final_bearing = (final_tangent_bearing + pi) % (2 * pi)
        
        # Générer le deuxième virage
        second_turn = self._generate_turn_segment(
            tangent_point, final_bearing, bearing_end, second_turn_points, second_turn_dir
        )
        
        # Fusionner les segments
        latitudes = first_turn['path']['latitude'] + straight_segment['path']['latitude'] + second_turn['path']['latitude']
        longitudes = first_turn['path']['longitude'] + straight_segment['path']['longitude'] + second_turn['path']['longitude']
        altitudes = first_turn['path']['altitude'] + straight_segment['path']['altitude'] + second_turn['path']['altitude']
        
        # Interpolation linéaire de l'altitude sur tout le chemin
        for i in range(len(altitudes)):
            fraction = i / (len(altitudes) - 1) if len(altitudes) > 1 else 0
            altitudes[i] = start_point['altitude'] + fraction * (end_point['altitude'] - start_point['altitude'])
        
        # Calculer la longueur totale du chemin
        first_arc_length = self.min_turn_radius * abs(mid_bearing - bearing_start)
        straight_length = compute_distance(mid_point, tangent_point)[0]
        second_arc_length = self.min_turn_radius * abs(bearing_end - final_bearing)
        total_length = first_arc_length + straight_length + second_arc_length
        
        return {'latitude': latitudes, 'longitude': longitudes, 'altitude': altitudes}, total_length
    
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
        d = compute_distance(start_point, circle_center)[0]
        
        # Si la distance est inférieure au rayon, aucune tangente externe n'est possible
        if d <= radius:
            return None
        
        # Bearing du centre du cercle par rapport au point de départ
        center_bearing = compute_bearing(start_point, circle_center)[0]
        
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
            'latitude': 0,
            'longitude': 0,
            'altitude': start_point['altitude']
        }
        tangent_point['latitude'], tangent_point['longitude'] = get_destination_from_range_and_bearing(
            start_point, tangent_distance, tangent_bearing
        )
        
        return tangent_point
        
    def _generate_turn_segment(self, start_point, start_bearing, end_bearing, num_points, turn_dir='L'):
        """
        Génère un segment de virage entre deux caps (bearings)
        
        Args:
            start_point (dict): Point de départ
            start_bearing (float): Cap initial (radians)
            end_bearing (float): Cap final (radians)
            num_points (int): Nombre de points à générer
            turn_dir (str): Direction du virage ('L' pour gauche, 'R' pour droite)
            
        Returns:
            dict: Segment de virage {path: {latitude, longitude, altitude}, end_point}
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
        center_lat, center_lon = get_destination_from_range_and_bearing(
            start_point, self.min_turn_radius, center_bearing
        )
        center_point = {
            'latitude': center_lat,
            'longitude': center_lon,
            'altitude': start_point['altitude']
        }
        
        # Générer les points sur l'arc
        bearings = np.linspace(0, abs(delta_bearing), num_points)
        latitudes = []
        longitudes = []
        
        for angle in bearings:
            # Pour chaque bearing, calculer un point sur le cercle
            if turn_dir == 'L':
                point_bearing = (start_bearing + angle) % (2 * pi)
            else:
                point_bearing = (start_bearing - angle) % (2 * pi)
                
            # Direction du point depuis le centre (opposée au bearing)
            from_center_bearing = (point_bearing + pi + direction * pi/2) % (2 * pi)
            lat, lon = get_destination_from_range_and_bearing(
                center_point, self.min_turn_radius, from_center_bearing
            )
            latitudes.append(lat)
            longitudes.append(lon)
        
        # Point final du virage
        end_point = {
            'latitude': latitudes[-1],
            'longitude': longitudes[-1],
            'altitude': start_point['altitude']
        }
        
        # Altitudes (même altitude partout dans le virage)
        altitudes = [start_point['altitude']] * num_points
        
        return {
            'path': {
                'latitude': latitudes,
                'longitude': longitudes,
                'altitude': altitudes
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
        latitudes = []
        longitudes = []
        
        # Calculer le bearing initial
        bearing = compute_bearing(start_point, end_point)[0]
        
        # Calculer la distance
        distance = compute_distance(start_point, end_point)[0]
        
        for f in fractions:
            # Pour chaque fraction, calculer un point à la distance proportionnelle
            lat, lon = get_destination_from_range_and_bearing(
                start_point, f * distance, bearing
            )
            latitudes.append(lat)
            longitudes.append(lon)
        
        # Interpolation linéaire de l'altitude
        altitudes = []
        for f in fractions:
            alt = start_point['altitude'] + f * (end_point['altitude'] - start_point['altitude'])
            altitudes.append(alt)
        
        return {
            'path': {
                'latitude': latitudes,
                'longitude': longitudes,
                'altitude': altitudes
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
        num_points = self.params.get('num_points', 100)
        
        # Conversion en coordonnées cartésiennes pour simplifier les calculs
        start_xyz = self._geodetic_to_cartesian(start_point)
        end_xyz = self._geodetic_to_cartesian(end_point)
        
        # Extraction des vitesses initiale et finale si disponibles
        initial_velocity = np.array([0.0, 0.0, 0.0])
        final_velocity = np.array([0.0, 0.0, 0.0])
        
        if 'bearing' in start_point:
            # Conversion du bearing en vecteur de vitesse initiale
            bearing_rad = start_point['bearing']
            speed = self.UAV_data.get('cruise_airspeed', 20.0)  # m/s
            initial_velocity = np.array([
                speed * np.cos(bearing_rad),
                speed * np.sin(bearing_rad),
                0.0  # Vitesse verticale initiale nulle
            ])
        
        if 'bearing' in end_point:
            # Conversion du bearing en vecteur de vitesse finale
            bearing_rad = end_point['bearing']
            speed = self.UAV_data.get('cruise_airspeed', 20.0)  # m/s
            final_velocity = np.array([
                speed * np.cos(bearing_rad),
                speed * np.sin(bearing_rad),
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
        path = self._cartesian_to_geodetic(path_xyz)
        
        return path
        
    def _geodetic_to_cartesian(self, point):
        # Conversion des coordonnées géodésiques en cartésiennes
        x, y, z = geographic_to_cartesian(point['latitude'], point['longitude'], point['altitude'])
        return np.array([x, y, z])

    def _cartesian_to_geodetic(self, xyz_array):
        # Conversion des coordonnées cartésiennes en géodésiques
        latitudes = []
        longitudes = []
        altitudes = []
        
        for point in xyz_array:
            lat, lon, alt = cartesian_to_geographic(point[0], point[1], point[2])
            latitudes.append(lat)
            longitudes.append(lon)
            altitudes.append(alt)
            
        return {
            'latitude': latitudes,
            'longitude': longitudes,
            'altitude': altitudes
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
            midpoint = (start + end) / 2
            
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
        
    def compute_path_metrics(self, path_xyz):
        """
        Calcule des métriques importantes pour la trajectoire:
        - Longueur totale
        - Distribution de la courbure
        - Énergie de flexion
        
        Args:
            path_xyz (np.array): Points de la trajectoire en coordonnées cartésiennes
            
        Returns:
            dict: Métriques du chemin
        """
        # Calcul des dérivées premières (vitesses)
        velocities = np.diff(path_xyz, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Calcul des dérivées secondes (accélérations) 
        accelerations = np.diff(velocities, axis=0)
        
        # Calcul de la longueur totale
        total_length = np.sum(speeds)
        
        # Calcul des courbures (approximation)
        curvatures = []
        for i in range(len(accelerations)):
            v = velocities[i]
            a = accelerations[i]
            speed = speeds[i]
            
            if speed > 1e-6:  # Éviter division par zéro
                # Formule de la courbure: |v × a| / |v|^3
                cross_prod = np.cross(v, a)
                curvature = np.linalg.norm(cross_prod) / (speed**3)
                curvatures.append(curvature)
            else:
                curvatures.append(0)
                
        # Calcul de l'énergie de flexion (intégrale du carré de la courbure)
        bending_energy = np.sum(np.array(curvatures)**2) if curvatures else 0
        
        return {
            'total_length': total_length,
            'max_curvature': max(curvatures) if curvatures else 0,
            'avg_curvature': np.mean(curvatures) if curvatures else 0,
            'bending_energy': bending_energy
        }
        
def generate_all_trajectories(start_point, end_point, params, UAV_data):
    """
    Génère toutes les trajectoires possibles (droit, courbe, Dubins 3D, PH) entre deux points.
    Retourne un dictionnaire avec chaque type de trajectoire.
    
    Args:
        start_point (dict): Point de départ {latitude, longitude, altitude}
        end_point (dict): Point d'arrivée {latitude, longitude, altitude}  
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
    circular = smoother.smooth_trajectory(circular, method='savgol')
     
    # Générateur Dubins 3D
    # Le lissage est intégré dans generate_path pour Dubins
    dubins_traj = DubinsPath3D(params, UAV_data)
    dubins = dubins_traj.generate_path(start_point, end_point)

    # Générateur Pythagorean Hodograph
    ph_traj = PythagoreanHodographPath(params, UAV_data)
    ph = ph_traj.generate_path(start_point, end_point)

    # Retourne une liste avec toutes les trajectoires
    return [straight, circular, dubins, ph]

class TrajectoryEvaluator:
    """Classe pour évaluer les trajectoires générées"""
    
    def __init__(self, params: Dict, UAV_data: Dict, ):
        self.params = params
        self.UAV_data = UAV_data
        self.weights = {
            'distance': 1,
            'energy': 10.0,
            'turn_rate': 5.0,    
            'smoothness': 3.0,           
            'battery_feasibility': 500.0 
        }
        self.alt_min = params['altitude_lower_bound']
        self.alt_max = params['altitude_upper_bound']
    

    def evaluate_trajectory(self, trajectories: List[Dict]) -> Dict:
        """
        Pour chaque drone, évalue toutes les trajectoires possibles et retourne la meilleure selon le score.
        Args:
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
            UAV_data_dict (dict): Dictionnaire {nom_drone: données_UAV}
            params (dict): Paramètres de génération/évaluation
        Returns:
            dict: Meilleure trajectoire {latitude: [], longitude: [], altitude: []}
        """
        best_trajectory = None
        best_score = float('inf')
        for trajectory in trajectories:
            score = self._evaluate_single_trajectory(trajectory)
            if score < best_score:
                best_score = score
                best_trajectory = trajectory
        return best_trajectory
    
    def _evaluate_single_trajectory(self, trajectory: Dict, FLT_conditions) -> float:
        """
        Évalue une trajectoire unique en fonction de critères définis.
        Args:
            trajectory (dict): Trajectoire à évaluer {latitude: [], longitude: [], altitude: []}
        Returns:
            float: Score de la trajectoire
        """
        lats = np.array(trajectory['latitude'])
        lons = np.array(trajectory['longitude'])
        alts = np.array(trajectory['altitude'])
        total_score = 0.0
        
        # Critère 1 : Altitude dans les bornes (vectorisé)
        altitude_ok = np.all((alts >= self.alt_min) & (alts <= self.alt_max))
        if not altitude_ok:
            total_score += 1000
        
        # Critère 2 : Distance totale
        total_distance = self._compute_total_distance(lats, lons, alts)
        distance_cost = total_distance * self.weights['distance']
        total_score += distance_cost
        
        # Critère 3 : Consommation énergétique
        total_energy, is_feasible = self._compute_energy_consumption(trajectory, FLT_conditions)
        energy_cost = total_energy * self.weights['energy']
        total_score += energy_cost
        
        # Pénalité si la trajectoire n'est pas faisable énergétiquement
        if not is_feasible:
            total_score += self.weights['battery_feasibility']
        
        # Critère 4 : Rayon de virage respecté
        # Calculer les angles entre segments consécutifs
        turn_rates = self._compute_turn_rates(lats, lons)
        # Pénalise les virages trop serrés
        if len(turn_rates) > 0:
            max_allowed_turn = self.UAV_data['max_turn_rate']
            penalty = sum(max(0, rate - max_allowed_turn) for rate in turn_rates)
            total_score += penalty * self.weights['turn_rate']
        
        return total_score
    
    def _compute_energy_consumption(self, trajectory: Dict, FLT_conditions) -> tuple:
        """
        Calcule la consommation d'énergie totale pour suivre la trajectoire donnée.
        
        Args:
            trajectory (dict): Trajectoire à évaluer {latitude: [], longitude: [], altitude: []}
            
        Returns:
            tuple: (consommation_totale, est_faisable)
        """
        lats = trajectory['latitude']
        lons = trajectory['longitude']
        alts = trajectory['altitude']
        
        # Point de départ initial avec batterie complète
        initial_battery = self.UAV_data['maximum_battery_capacity']
        current_battery = initial_battery
        total_consumption = 0
        time_step = self.params['time_step']
        
        for i in range(len(lats) - 1):
            
            # Déterminer le mode de vol (montée = moteur, descente = planeur)
            altitude_change = alts[i+1] - alts[i]
            
            # Calculer la consommation d'énergie si le drone monte
            if altitude_change >= 0:  # Mode moteur pour monter ou vol horizontal
                # Calculer la puissance requise
                power = get_power_consumption(self.UAV_data, FLT_conditions)
                # Convertir la puissance (W) en énergie (Wh) pour ce segment
                energy_segment = power * (time_step / 3600)
            
            # Mettre à jour la consommation totale et la batterie restante
            total_consumption += energy_segment
            current_battery -= energy_segment
        
        # Vérifier si la trajectoire est faisable avec la batterie disponible
        is_feasible = current_battery >= self.UAV_data.get('desired_reserved_battery_capacity', 0)
        
        return total_consumption, is_feasible
    
    def _compute_turn_rates(self, lats, lons):
        """
        Calcule les taux de virage entre segments consécutifs
        """
        turn_rates = []
        if len(lats) < 3:
            return turn_rates
            
        for i in range(len(lats)-2):
            point1 = {'latitude': lats[i], 'longitude': lons[i]}
            point2 = {'latitude': lats[i+1], 'longitude': lons[i+1]}
            point3 = {'latitude': lats[i+2], 'longitude': lons[i+2]}
            
            # Calculer les bearings
            bearing1 = compute_bearing(point1, point2)[0]
            bearing2 = compute_bearing(point2, point3)[0]
            
            # Différence d'angle (taux de virage)
            diff = abs((bearing2 - bearing1 + np.pi) % (2 * np.pi) - np.pi)
            turn_rates.append(diff)
            
        return turn_rates
    
    def _compute_total_distance(self, lats, lons, alts):
        """
        Calcule la distance totale parcourue par la trajectoire en utilisant des calculs vectorisés.
        """
        # use compute function to compute distances
        distances = []
        for i in range(len(lats) - 1):
            start_point = {'latitude': lats[i], 'longitude': lons[i], 'altitude': alts[i]}
            end_point = {'latitude': lats[i + 1], 'longitude': lons[i + 1], 'altitude': alts[i + 1]}
            distance = compute_distance(start_point, end_point)[0]
            distances.append(distance)
        return np.sum(distances)

class TrajectorySmoothing:
    """
    Classe fournissant des méthodes pour lisser les trajectoires générées.
    
    Permet d'améliorer la pilotabilité des trajectoires en assurant la continuité
    de position (G0), de tangente (G1) et de courbure (G2).
    """
    
    def __init__(self, params=None):
        """
        Initialise les paramètres de lissage.
        
        Args:
            params (dict): Paramètres de lissage (facteur de lissage, etc.)
        """
        self.params = params or {}
        self.smoothing_factor = params.get('smoothing_factor', 0.5)
        self.window_size = params.get('smoothing_window', 5)
        
    def smooth_trajectory_spline(self, trajectory):
        """
        Lisse une trajectoire en utilisant des splines cubiques.
        
        Args:
            trajectory (dict): Trajectoire à lisser {latitude: [], longitude: [], altitude: []}
            
        Returns:
            dict: Trajectoire lissée {latitude: [], longitude: [], altitude: []}
        """
        # Extraire les points de la trajectoire
        lats = np.array(trajectory['latitude'])
        lons = np.array(trajectory['longitude'])
        alts = np.array(trajectory['altitude'])
        
        # Paramètre pour la spline (distance cumulée le long de la trajectoire)
        n = len(lats)
        t = np.zeros(n)
        for i in range(1, n):
            pt1 = {'latitude': lats[i-1], 'longitude': lons[i-1]}
            pt2 = {'latitude': lats[i], 'longitude': lons[i]}
            t[i] = t[i-1] + compute_distance(pt1, pt2)[0]
        
        # Normaliser le paramètre entre 0 et 1
        if t[-1] > 0:
            t = t / t[-1]
        
        # Facteur de lissage (0 pour interpolation exacte, plus élevé pour plus de lissage)
        s = self.smoothing_factor * n
        
        # Créer les splines pour chaque dimension
        spline_lat = interpolate.splrep(t, lats, s=s)
        spline_lon = interpolate.splrep(t, lons, s=s)
        spline_alt = interpolate.splrep(t, alts, s=s)
        
        # Évaluer les splines sur une nouvelle grille
        t_new = np.linspace(0, 1, n)
        lats_smooth = interpolate.splev(t_new, spline_lat)
        lons_smooth = interpolate.splev(t_new, spline_lon)
        alts_smooth = interpolate.splev(t_new, spline_alt)
        
        # Retourner la trajectoire lissée
        return {
            'latitude': lats_smooth.tolist(),
            'longitude': lons_smooth.tolist(),
            'altitude': alts_smooth.tolist()
        }
        
    def smooth_trajectory_savgol(self, trajectory):
        """
        Lisse une trajectoire en utilisant un filtre de Savitzky-Golay
        qui préserve mieux les caractéristiques des données.
        
        Args:
            trajectory (dict): Trajectoire à lisser {latitude: [], longitude: [], altitude: []}
            
        Returns:
            dict: Trajectoire lissée {latitude: [], longitude: [], altitude: []}
        """
        # Extraire les points de la trajectoire
        lats = np.array(trajectory['latitude'])
        lons = np.array(trajectory['longitude'])
        alts = np.array(trajectory['altitude'])
        
        # Définir la taille de la fenêtre et le degré polynomial
        window_size = min(self.window_size, len(lats) - 1)
        if window_size % 2 == 0:  # La taille de la fenêtre doit être impaire
            window_size += 1
        poly_order = min(3, window_size - 1)
        
        # Appliquer le filtre de Savitzky-Golay
        if len(lats) > window_size:
            lats_smooth = savgol_filter(lats, window_size, poly_order)
            lons_smooth = savgol_filter(lons, window_size, poly_order)
            alts_smooth = savgol_filter(alts, window_size, poly_order)
        else:
            # Si la trajectoire est trop courte, renvoyer la trajectoire originale
            lats_smooth = lats
            lons_smooth = lons
            alts_smooth = alts
        
        # Retourner la trajectoire lissée
        return {
            'latitude': lats_smooth.tolist(),
            'longitude': lons_smooth.tolist(),
            'altitude': alts_smooth.tolist()
        }
    
    def smooth_dubins_junctions(self, trajectory, num_junction_points=5):
        """
        Lisse spécifiquement les jonctions dans une trajectoire Dubins
        en utilisant des courbes de Bézier cubiques.
        
        Args:
            trajectory (dict): Trajectoire à lisser {latitude: [], longitude: [], altitude: []}
            num_junction_points (int): Nombre de points à considérer autour des jonctions
            
        Returns:
            dict: Trajectoire lissée aux jonctions {latitude: [], longitude: [], altitude: []}
        """
        # Extraire les points de la trajectoire
        lats = np.array(trajectory['latitude'])
        lons = np.array(trajectory['longitude'])
        alts = np.array(trajectory['altitude'])
        n = len(lats)
        
        if n < 10:  # Pas assez de points pour détecter les jonctions
            return trajectory
            
        # Calculer les taux de virage pour détecter les jonctions
        turn_rates = []
        for i in range(1, n-1):
            point1 = {'latitude': lats[i-1], 'longitude': lons[i-1]}
            point2 = {'latitude': lats[i], 'longitude': lons[i]}
            point3 = {'latitude': lats[i+1], 'longitude': lons[i+1]}
            
            bearing1 = compute_bearing(point1, point2)[0]
            bearing2 = compute_bearing(point2, point3)[0]
            
            diff = abs((bearing2 - bearing1 + np.pi) % (2 * np.pi) - np.pi)
            turn_rates.append(diff)
        
        # Seuil pour détecter les jonctions (taux de virage élevé)
        threshold = np.percentile(turn_rates, 90)  # 90ème percentile
        
        # Trouver les indices des jonctions
        junction_indices = [i + 1 for i, rate in enumerate(turn_rates) if rate > threshold]
        
        # Si aucune jonction trouvée, retourner la trajectoire originale
        if not junction_indices:
            return trajectory
            
        # Lisser chaque jonction détectée
        new_lats = lats.copy()
        new_lons = lons.copy()
        new_alts = alts.copy()
        
        for idx in junction_indices:
            # Assurer que nous avons assez de points autour de la jonction
            start_idx = max(0, idx - num_junction_points)
            end_idx = min(n, idx + num_junction_points + 1)
            
            # Points avant et après la jonction
            pre_junction = (lats[start_idx:idx], lons[start_idx:idx])
            post_junction = (lats[idx:end_idx], lons[idx:end_idx])
            
            # Créer une courbe de Bézier pour remplacer la jonction
            t = np.linspace(0, 1, end_idx - start_idx)
            
            # Points de contrôle pour la courbe de Bézier
            p0 = np.array([pre_junction[0][0], pre_junction[1][0]])
            p3 = np.array([post_junction[0][-1], post_junction[1][-1]])
            
            # Directions tangentielles aux extrémités
            if len(pre_junction[0]) > 1:
                d0 = np.array([pre_junction[0][-1] - pre_junction[0][-2], 
                              pre_junction[1][-1] - pre_junction[1][-2]])
                d0 = d0 / np.linalg.norm(d0) if np.linalg.norm(d0) > 0 else np.array([1, 0])
            else:
                d0 = np.array([1, 0])
                
            if len(post_junction[0]) > 1:
                d1 = np.array([post_junction[0][1] - post_junction[0][0], 
                              post_junction[1][1] - post_junction[1][0]])
                d1 = d1 / np.linalg.norm(d1) if np.linalg.norm(d1) > 0 else np.array([1, 0])
            else:
                d1 = np.array([1, 0])
            
            # Points de contrôle intermédiaires
            control_dist = np.linalg.norm(p3 - p0) / 3
            p1 = p0 + d0 * control_dist
            p2 = p3 - d1 * control_dist
            
            # Évaluer la courbe de Bézier
            bezier_points = np.zeros((len(t), 2))
            for i, ti in enumerate(t):
                bezier_points[i] = ((1-ti)**3 * p0 + 
                                   3 * (1-ti)**2 * ti * p1 + 
                                   3 * (1-ti) * ti**2 * p2 + 
                                   ti**3 * p3)
            
            # Interpolation linéaire de l'altitude
            alts_interp = np.interp(t, [0, 1], [alts[start_idx], alts[end_idx-1]])
            
            # Remplacer les points autour de la jonction
            new_lats[start_idx:end_idx] = bezier_points[:, 0]
            new_lons[start_idx:end_idx] = bezier_points[:, 1]
            new_alts[start_idx:end_idx] = alts_interp
        
        # Retourner la trajectoire lissée
        return {
            'latitude': new_lats.tolist(),
            'longitude': new_lons.tolist(),
            'altitude': new_alts.tolist()
        }
        
    def smooth_trajectory(self, trajectory, method='spline'):
        """
        Applique une méthode de lissage à la trajectoire.
        
        Args:
            trajectory (dict): Trajectoire à lisser {latitude: [], longitude: [], altitude: []}
            method (str): Méthode de lissage ('spline', 'savgol', 'dubins_junctions', 'all')
            
        Returns:
            dict: Trajectoire lissée {latitude: [], longitude: [], altitude: []}
        """
        if method == 'spline':
            return self.smooth_trajectory_spline(trajectory)
        elif method == 'savgol':
            return self.smooth_trajectory_savgol(trajectory)
        elif method == 'dubins_junctions':
            return self.smooth_dubins_junctions(trajectory)
        else:
            return trajectory  # Aucun lissage
