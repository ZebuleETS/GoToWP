import numpy as np
from math import pi
from abc import ABC, abstractmethod
from compute import (
    cartesian_to_geographic,
    compute_bearing,
    compute_distance,
    geographic_to_cartesian,
    get_destination_from_range_and_bearing
)
from typing import Dict

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
        num_points = self.params.get('num_points', 100)
        
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
            latitudes.append(dest['latitude'])
            longitudes.append(dest['longitude'])
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

    def generate_path(self, start_point, end_point):
        """
        Génère une trajectoire circulaire géodésiques entre deux points en utilisant les fonctions utilitaires.
        Args:
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
            params (dict): Paramètres UAV et simulation 
        Returns:
            dict: Points de trajectoire {latitude: [], longitude: [], altitude: []}
        """
        num_points = self.params.get('num_points', 100)
        radius = self.params.get('radius', 100)  # rayon en mètres

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
    Args:
            params (dict): Parameters for the trajectory generation, e.g., number of points, radius, etc.
            UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    
    def __init__(self):
        super().__init__(params=None, UAV_data=None)
        self.min_turn_radius = 30  # rayon de virage minimum en mètres

    def generate_path(self, start_point, end_point):
        """
        Génère une trajectoire Dubins 3D entre deux points avec deux virages et un segment droit.
        Args:
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
            params (dict): Paramètres UAV
        Returns:
            dict: Points de trajectoire {latitude: [], longitude: [], altitude: []}
        """
        num_points = self.params.get('num_points', 100)
        
        LBz = self.params['altitude_lower_bound']
        UBz = self.params['altitude_upper_bound']
        max_turn_rate = self.UAV_data['max_turn_rate']
        min_velocity = self.UAV_data['min_airspeed']
        max_velocity = self.UAV_data['max_airspeed']
        time_step = self.params['time_step']
        horizon_length = self.params['horizon_length']
        
        num_steps = int(horizon_length / time_step)
        
        # Calcul du rayon de virage minimum
        avg_velocity = (min_velocity + max_velocity) / 2
        min_turn_radius = avg_velocity / max_turn_rate
        
        x_start, y_start, z_start = geographic_to_cartesian(
            start_point['latitude'], start_point['longitude'], start_point['altitude']
        )
        x_end, y_end, z_end = geographic_to_cartesian(
            end_point['latitude'], end_point['longitude'], end_point['altitude']
        )
        
        
        
        latitudes = []
        longitudes = []
        altitudes = []
        
        return {'latitude': latitudes, 'longitude': longitudes, 'altitude': altitudes.tolist()}
        
    def _generate_turn(self, start_point, heading, num_points):
        # Utilise CircularTrajectory pour générer un virage
        params = {
            'num_points': num_points,
            'radius': self.min_turn_radius
        }
        # Le point d'arrivée du virage est calculé à une distance d'un demi-cercle
        arc_length = pi * self.min_turn_radius  # demi-cercle
        end_point_lat, end_point_lon = get_destination_from_range_and_bearing(
            start_point, arc_length, heading
        )
        end_point = {
            'latitude': end_point_lat,
            'longitude': end_point_lon,
            'altitude': start_point['altitude']
        }
        circular_traj = CircularTrajectory(params, self.UAV_data)
        path = circular_traj.generate_path(start_point, end_point, params)
        return {'path': path, 'end_point': end_point}

    def _generate_straight_segment(self, start_point, end_point, num_points):
        # Utilise StraightLineTrajectory pour générer un segment droit
        params = {
            'num_points': num_points
        }
        straight_traj = StraightLineTrajectory(params, self.UAV_data)
        path = straight_traj.generate_path(start_point, end_point, None, None, None, params)
        return {'path': path, 'end_point': end_point}

class PythagoreanHodographPath(TrajectoryGenerator):
    """
    Pythagorean Hodograph path trajectory generator
    Args:
            params (dict): Parameters for the trajectory generation, e.g., number of points, radius, etc.
            UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)
    
    def generate_path(self, start_point, end_point):
        num_points = self.params.get('num_points', 100)
        
        # Conversion en coordonnées cartésiennes pour simplifier les calculs
        start_xyz = self._geodetic_to_cartesian(start_point)
        end_xyz = self._geodetic_to_cartesian(end_point)
        
        # Génération des points de contrôle pour la courbe PH
        control_points = self._generate_control_points(start_xyz, end_xyz)
        
        # Génération de la courbe PH
        t = np.linspace(0, 1, num_points)
        path_xyz = self._evaluate_ph_curve(control_points, t)
        
        # Conversion retour en coordonnées géodésiques
        path = self._cartesian_to_geodetic(path_xyz)
        
        return path
        
    def _geodetic_to_cartesian(self, point):
        # Conversion des coordonnées géodésiques en cartésiennes
        x, y, z = geographic_to_cartesian(point['latitude'], point['longitude'], point['altitude'])
        return np.array([x, y, z])

    def _cartesian_to_geodetic(self, xyz):
        # Conversion des coordonnées cartésiennes en géodésiques
        x, y, z = xyz
        return cartesian_to_geographic(x, y, z)

    def _generate_control_points(self, start, end):
        # Génération des points de contrôle pour la courbe PH
        pass  # À implémenter
        
    def _evaluate_ph_curve(self, control_points, t):
        # Évaluation de la courbe PH
        pass  # À implémenter

def generate_all_trajectories(start_point, end_point, UAV_data, params):
    """
    Génère toutes les trajectoires possibles (droit, courbe, Dubins 3D, PH) entre deux points.
    Retourne un dictionnaire avec chaque type de trajectoire.
    """
    # Générateur ligne droite
    straight_traj = StraightLineTrajectory(params, UAV_data)
    straight = straight_traj.generate_path(start_point, end_point)

    # Générateur courbe circulaire
    circular_traj = CircularTrajectory(params, UAV_data)
    circular = circular_traj.generate_path(start_point, end_point)

    # Générateur Dubins 3D
    dubins_traj = DubinsPath3D(params, UAV_data)
    # Pour Dubins, il faut fournir les headings (cap) de départ/arrivée
    dubins = dubins_traj.generate_path(start_point, end_point)

    # Générateur Pythagorean Hodograph
    ph_traj = PythagoreanHodographPath(params, UAV_data)
    ph = ph_traj.generate_path(start_point, end_point)

    return {
        'straight': straight,
        'circular': circular,
        'dubins': dubins,
        'pythagorean_hodograph': ph
    }

class TrajectoryEvaluator:
    """Classe pour évaluer les trajectoires générées"""
    
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data

    def evaluate_trajectory(self, trajectories: Dict) -> Dict:
        """
        Pour chaque drone, évalue toutes les trajectoires possibles et retourne la meilleure selon le score.
        Args:
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
            UAV_data_dict (dict): Dictionnaire {nom_drone: données_UAV}
            params (dict): Paramètres de génération/évaluation
        Returns:
            dict: {'trajectory': trajectoire}
        """
        best_trajectory = None
        best_score = float('inf')
        for name, trajectory in trajectories.items():
            score = self._evaluate_single_trajectory(trajectory)
            if score < best_score:
                best_score = score
                best_trajectory = trajectory
        return {'trajectory': best_trajectory}
    
    def _evaluate_single_trajectory(self, trajectory: Dict) -> float:
        """
        Évalue une trajectoire unique en fonction de critères définis.
        Args:
            trajectory (dict): Trajectoire à évaluer {latitude: [], longitude: [], altitude: []}
        Returns:
            float: Score de la trajectoire
        """
        # Critère 1 : Altitude dans les bornes (vectorisé)
        alts = np.array(trajectory['altitude'])
        altitude_ok = np.all((alts >= self.alt_min) & (alts <= self.alt_max))
        
        # Critère 2 : Distance totale (optimisée)
        
        # Critère 3 : Consommation énergétique
        
        # Critère 4 : Rayon de virage minimum
        
        # Critère 5 : Vitesse moyenne réaliste
        
        # Exemple d'évaluation basée sur la distance totale parcourue
        total_distance = 0.0
        for i in range(1, len(trajectory['latitude'])):
            dist = compute_distance(
                {'latitude': trajectory['latitude'][i-1], 'longitude': trajectory['longitude'][i-1]},
                {'latitude': trajectory['latitude'][i], 'longitude': trajectory['longitude'][i]}
            )[0]
            total_distance += dist
        
        # Ajouter d'autres critères d'évaluation si nécessaire
        return total_distance
