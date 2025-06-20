import numpy as np
from math import pi, sqrt
from scipy.interpolate import CubicSpline
from abc import ABC, abstractmethod
from compute import (
    cartesian_to_geographic,
    geographic_to_cartesian,
    get_power_consumption,
    get_sink_rate,
    get_destination_from_range_and_bearing
)
from typing import Dict, Tuple

n_combinations = None

class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators"""
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data

    @abstractmethod
    def generate_path(self, start_point, end_point, FLT_data, Uidx, candidate_positions, params):
        """
        Generates a trajectory between two points
        
        Args:
            start_point (dict): Starting point {latitude, longitude, altitude}
            end_point (dict): End point {latitude, longitude, altitude}
            FLT_data (dict): Flight data for the UAV
            Uidx (int): Index of the UAV
            candidate_positions (dict): Candidate positions for the trajectory
            params (dict): Additional method-specific parameters
            
        Returns:
            dict: Trajectory points {latitude: [], longitude: [], altitude: []}
        """
        pass

class TrajectoryCalculator:
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data
        
    def generate_trajectory_options(self, FLT_data: Dict, Uidx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Génère les options de trajectoire vectorisées"""
        # Création des grilles de cap et vitesse
        h_step = self.params['bearing_step']
        v_step = self.params['speed_step']
        max_turn_rate = self.UAV_data['max_turn_rate']
        min_velocity = self.UAV_data['min_airspeed']
        max_velocity = self.UAV_data['max_airspeed']
        
        Hr = np.linspace(FLT_data[Uidx]['bearing'], 
                        FLT_data[Uidx]['bearing'] + max_turn_rate, h_step)
        Hl = np.linspace(FLT_data[Uidx]['bearing'] - max_turn_rate,
                        FLT_data[Uidx]['bearing'], h_step)
        H = np.hstack([Hl, Hr[1:]])
        
        Vr = np.linspace(FLT_data[Uidx]['airspeed'], max_velocity, v_step)
        Vl = np.linspace(min_velocity, FLT_data[Uidx]['airspeed'], v_step)
        V = np.hstack([Vl, Vr[1:]])
        
        return np.meshgrid(H, V)

    def calculate_trajectories(self, FLT_data: Dict, Uidx: int, mode: str) -> Dict:
        """Calcule les trajectoires possibles selon le mode de vol"""
        # Génération des grilles de cap et vitesse
        H_grid, V_grid = self.generate_trajectory_options(FLT_data, Uidx)
        n_combinations = H_grid.size
        
        # Configuration des paramètres
        t_step = self.params['time_step']
        LBz = self.params['altitude_lower_bound']
        UBz = self.params['altitude_upper_bound']
        
        # Initialisation des résultats potentiels
        candidate_positions = {
            'latitude': np.zeros(n_combinations),
            'longitude': np.zeros(n_combinations),
            'altitude': np.zeros(n_combinations),
            'bearing': H_grid.flatten(),
            'airspeed': V_grid.flatten(),
            'battery_capacity': np.zeros(n_combinations),
            'flight_path_angle': np.zeros(n_combinations),
            'flight_mode': np.array([mode] * n_combinations)
        }
        
        if mode == 'glide':
            return self._calculate_glide_trajectories(
                FLT_data, Uidx, candidate_positions, t_step, LBz, UBz
            )
        else: # mode == 'engine'
            return self._calculate_engine_trajectories(
                FLT_data, Uidx, candidate_positions, t_step, LBz, UBz
            )
    
    def _calculate_glide_trajectories(self, FLT_data, Uidx, candidate_positions, 
                                    t_step, LBz, UBz):
        """Calcul vectorisé des trajectoires en mode planeur"""
        airspeeds = candidate_positions['airspeed']
        
        # Calcul vectorisé du taux de descente
        FLT_conditions_vec = {
            'airspeed': airspeeds,
            'airspeed_dot': np.zeros_like(airspeeds)
        }
        sink_rates = np.vectorize(get_sink_rate)(self.UAV_data, FLT_conditions_vec)
        
        # Calcul des déplacements
        dZ = -sink_rates * t_step
        TD = airspeeds * t_step
        
        # Calcul des nouvelles positions
        current_pos = {
            'latitude': FLT_data[Uidx]['latitude'],
            'longitude': FLT_data[Uidx]['longitude']
        }
        
        # Calcul vectorisé des destinations
        lats, lons = np.vectorize(get_destination_from_range_and_bearing)(
            [current_pos] * len(TD), 
            TD,
            candidate_positions['bearing']
        )
        
        # Mise à jour des positions
        candidate_positions['latitude'] = lats
        candidate_positions['longitude'] = lons
        candidate_positions['altitude'] = np.clip(
            FLT_data[Uidx]['altitude'] + dZ,
            LBz,
            UBz
        )
        candidate_positions['battery_capacity'].fill(FLT_data[Uidx]['battery_capacity'])
        
        return candidate_positions
    
    def _calculate_engine_trajectories(self, FLT_data, Uidx, candidate_positions, 
                                     t_step, LBz, UBz):
        """Calcul vectorisé des trajectoires en mode moteur"""
        airspeeds = candidate_positions['airspeed']
        
        # Calcul des déplacements
        dZ = airspeeds * t_step
        
        # Calcul de la consommation d'énergie
        FLT_conditions_vec = {
            'airspeed': airspeeds,
            'airspeed_dot': np.zeros_like(airspeeds)
        }
        power = np.vectorize(get_power_consumption)(self.UAV_data, FLT_conditions_vec)
        power_consumption = power * (t_step / 3600)
        
        # Calcul des nouvelles positions
        current_pos = {
            'latitude': FLT_data[Uidx]['latitude'],
            'longitude': FLT_data[Uidx]['longitude']
        }
        
        # Calcul vectorisé des destinations
        lats, lons = np.vectorize(get_destination_from_range_and_bearing)(
            [current_pos] * len(dZ), 
            dZ,
            candidate_positions['bearing']
        )
        
        # Mise à jour des positions
        candidate_positions['latitude'] = lats
        candidate_positions['longitude'] = lons
        candidate_positions['altitude'] = np.clip(
            FLT_data[Uidx]['altitude'] + dZ,
            LBz,
            UBz
        )
        candidate_positions['battery_capacity'] = (
            FLT_data[Uidx]['battery_capacity'] - power_consumption
        )
        
        return candidate_positions
    

class StraightLineTrajectory(TrajectoryGenerator):
    """Straight line trajectory generator"""
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)


    def generate_path(self, start_point, end_point, FLT_data, Uidx, candidate_positions, params):
        """
        Génère une trajectoire en ligne droite entre deux points
        
        Args:
            start_point (dict): Point de départ {latitude, longitude, altitude}
            end_point (dict): Point d'arrivée {latitude, longitude, altitude}
            params (dict): Paramètres UAV et simulation 
            
        Returns:
            dict: Points de trajectoire {latitude: [], longitude: [], altitude: []}
        """
        num_points = params.get('num_points', 100)
        LBz = self.params['altitude_lower_bound']
        UBz = self.params['altitude_upper_bound']
        
        # Interpolation linéaire entre les points
        lat = np.linspace(start_point['latitude'], end_point['latitude'], num_points)
        lon = np.linspace(start_point['longitude'], end_point['longitude'], num_points)
        alt = np.linspace(start_point['altitude'], end_point['altitude'], num_points)
        
        return {'latitude': lat.tolist(), 'longitude': lon.tolist(), 'altitude': alt.tolist()}

class CircularTrajectory(TrajectoryGenerator):
    """Circular trajectory generator"""
    
    def generate_path(self, start_point, end_point, params):
        num_points = params.get('num_points', 100)
        radius = params.get('radius', 100)  # rayon en mètres
        
        # Calcul du centre du cercle
        center_lat = (start_point['latitude'] + end_point['latitude']) / 2
        center_lon = (start_point['longitude'] + end_point['longitude']) / 2
        
        # Génération des points sur le cercle
        angles = np.linspace(0, 2*pi, num_points)
        lat = center_lat + (radius/111319.9) * np.cos(angles)  # conversion approximative en degrés
        lon = center_lon + (radius/(111319.9*np.cos(center_lat))) * np.sin(angles)
        
        # Interpolation de l'altitude
        alt = np.linspace(start_point['altitude'], end_point['altitude'], num_points)
        
        return {'latitude': lat.tolist(), 'longitude': lon.tolist(), 'altitude': alt.tolist()}

class DubinsPath3D(TrajectoryGenerator):
    """3D Dubins path trajectory generator"""
    
    def __init__(self):
        super().__init__(params=None, UAV_data=None)
        self.params = {'num_points': 100}
        self.min_turn_radius = 30  # rayon de virage minimum en mètres

    def generate_path(self, start_point, end_point, params):
        """
        Génère une trajectoire Dubins 3D optimisée
        
        Args:
            start_point (dict): Point de départ avec format FLT_data (inclut bearing)
            end_point (dict): Point d'arrivée avec heading optionnel
            params (dict): Paramètres UAV
            
        Returns:
            dict: Solution candidate complète
        """
        
        # 1. Premier virage (départ)
        turn1 = self._generate_turn(start_point, params['start_heading'], params['num_points']//3)
        
        # 2. Segment droit
        straight = self._generate_straight_segment(
            turn1['end_point'],
            {'latitude': end_point['latitude'], 'longitude': end_point['longitude'], 
             'altitude': end_point['altitude']},
            params['num_points']//3
        )
        
        # 3. Dernier virage (arrivée)
        turn2 = self._generate_turn(straight['end_point'], params['end_heading'], params['num_points']//3)
        
        # Combinaison des segments
        return {
            'latitude': turn1['path']['latitude'] + straight['path']['latitude'] + turn2['path']['latitude'],
            'longitude': turn1['path']['longitude'] + straight['path']['longitude'] + turn2['path']['longitude'],
            'altitude': turn1['path']['altitude'] + straight['path']['altitude'] + turn2['path']['altitude']
        }
        
    def _generate_turn(self, start_point, heading, num_points):
        # Génération d'un virage avec le rayon minimum
        pass  # À implémenter avec la géométrie différentielle
        #TODO finish this method
    def _generate_straight_segment(self, start_point, end_point, num_points):
        # Génération d'un segment droit
        path = StraightLineTrajectory.generate_segment(start_point, end_point, num_points)
        return {'path': path}

class PythagoreanHodographPath(TrajectoryGenerator):
    """Spatial Pythagorean hodograph trajectory generator"""
    
    def generate_path(self, start_point, end_point, params):
        num_points = params.get('num_points', 100)
        
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

def smooth_trajectory(trajectory, smoothing_factor=0.5):    
    """
    Smooths a trajectory using cubic spline interpolation
    
    Args:
        trajectory (dict): Trajectory to smooth {latitude: [], longitude: [], altitude: []}
        smoothing_factor (float): Smoothing factor between 0 and 1 
        
    Returns:
        dict: Smoothed trajectory
    """
    num_points = len(trajectory['latitude'])
    t = np.linspace(0, 1, num_points)
    
    # Création des splines pour chaque dimension
    cs_lat = CubicSpline(t, trajectory['latitude'], bc_type='natural')
    cs_lon = CubicSpline(t, trajectory['longitude'], bc_type='natural')
    cs_alt = CubicSpline(t, trajectory['altitude'], bc_type='natural')
    
    # Évaluation des splines avec plus de points pour un lissage plus fin
    t_new = np.linspace(0, 1, int(num_points/smoothing_factor))
    
    return {
        'latitude': cs_lat(t_new).tolist(),
        'longitude': cs_lon(t_new).tolist(),
        'altitude': cs_alt(t_new).tolist()
    }

def compute_path_metrics(trajectory):    
    """
    Computes various metrics for a trajectory
    
    Args:
        trajectory (dict): Trajectory to analyze
        
    Returns:
        dict: Computed metrics including:
            - length: total path length
            - curvature: curvature at each point
            - torsion: torsion at each point
            - smoothness: measure of trajectory regularity
    """
    metrics = {
        'length': 0,  # longueur totale
        'curvature': [],  # courbure en chaque point
        'torsion': [],  # torsion en chaque point
        'smoothness': 0,  # mesure de la régularité
    }
    
    # Calcul de la longueur
    for i in range(len(trajectory['latitude'])-1):
        d_lat = trajectory['latitude'][i+1] - trajectory['latitude'][i]
        d_lon = trajectory['longitude'][i+1] - trajectory['longitude'][i]
        d_alt = trajectory['altitude'][i+1] - trajectory['altitude'][i]
        
        metrics['length'] += sqrt(d_lat**2 + d_lon**2 + d_alt**2)
    
    # TODO: Implémenter les calculs de courbure et torsion
    
    return metrics

class TrajectoryEvaluator:
    """Evaluates and compares different trajectory options"""

    def __init__(self, uav_data, flight_conditions, safety_params):
        """
        Initialize the evaluator with UAV and environment parameters
        
        Args:
            uav_data (dict): UAV physical parameters
            flight_conditions (dict): Current flight conditions
            safety_params (dict): Safety parameters (min distances, etc.)
        """
        self.uav_data = uav_data
        self.flight_conditions = flight_conditions
        self.safety_params = safety_params
        
        # Initialize trajectory generators
        self.generators = {
            'straight': StraightLineTrajectory(),
            'circular': CircularTrajectory(),
            'dubins': DubinsPath3D(),
            'ph': PythagoreanHodographPath()
        }

    def evaluate_all_trajectories(self, start_point, end_point, obstacles=None):
        """
        Evaluates all possible trajectory types between two points
        
        Args:
            start_point (dict): Starting position {latitude, longitude, altitude}
            end_point (dict): End position {latitude, longitude, altitude}
            obstacles (list, optional): List of obstacle positions to avoid
            
        Returns:
            dict: Best trajectory and its scores
        """
        trajectories = {}
        scores = {}
        
        # Generate trajectories using each method
        params = {
            'num_points': 100,
            'radius': self.uav_data.get('min_turn_radius', 30),
            'start_heading': self.flight_conditions.get('bearing', 0),
            'end_heading': None  # Will be calculated based on end point
        }
        
        for name, generator in self.generators.items():
            try:
                trajectory = generator.generate_path(start_point, end_point, params)
                trajectories[name] = trajectory
                scores[name] = self._evaluate_trajectory(trajectory, obstacles)
            except Exception as e:
                print(f"Warning: Failed to generate {name} trajectory: {str(e)}")
                continue

        # Find the best trajectory based on weighted scores
        best_trajectory_name = self._select_best_trajectory(scores)
        best_trajectory = trajectories[best_trajectory_name]
        
        return {
            'trajectory': best_trajectory,
            'type': best_trajectory_name,
            'scores': scores[best_trajectory_name],
            'all_scores': scores
        }

    def _evaluate_trajectory(self, trajectory, obstacles=None):
        """
        Evaluates a single trajectory based on multiple criteria
        
        Args:
            trajectory (dict): Trajectory to evaluate
            obstacles (list, optional): List of obstacle positions to avoid
            
        Returns:
            dict: Scores for different criteria
        """
        # Get basic metrics
        metrics = compute_path_metrics(trajectory)
        
        # Calculate scores for different criteria
        scores = {
            'length': self._score_length(metrics['length']),
            'smoothness': self._score_smoothness(trajectory),
            'energy': self._score_energy_consumption(trajectory),
            'safety': self._score_safety(trajectory, obstacles) if obstacles else 1.0
        }
        
        # Calculate final weighted score
        scores['total'] = self._compute_weighted_score(scores)
        
        return scores

    def _score_length(self, length):
        """Scores trajectory based on its length (shorter is better)"""
        # Normalize against a reference maximum length
        max_acceptable_length = self.safety_params.get('max_path_length', 5000)  # meters
        return max(0, 1 - length/max_acceptable_length)

    def _score_smoothness(self, trajectory):
        """Scores trajectory based on its smoothness"""
        # Calculate rate of change of heading
        headings = self._compute_headings(trajectory)
        heading_changes = np.diff(headings)
        
        # Normalize heading changes
        max_acceptable_change = np.radians(self.uav_data.get('max_turn_rate', 45))  # degrees/s
        smoothness = 1 - np.mean(np.abs(heading_changes)) / max_acceptable_change
        
        return max(0, min(1, smoothness))

    def _score_energy_consumption(self, trajectory):
        """Scores trajectory based on estimated energy consumption"""
        # Estimate energy based on path length and altitude changes
        total_distance = compute_path_metrics(trajectory)['length']
        altitude_changes = np.diff(trajectory['altitude'])
        
        # Energy cost factors
        horizontal_cost = total_distance * self.uav_data.get('horizontal_energy_factor', 1)
        vertical_cost = np.sum(np.abs(altitude_changes)) * self.uav_data.get('vertical_energy_factor', 2)
        
        total_energy_cost = horizontal_cost + vertical_cost
        max_energy = self.flight_conditions.get('battery_capacity', 1000)  # Wh
        
        return max(0, 1 - total_energy_cost/max_energy)

    def _score_safety(self, trajectory, obstacles):
        """Scores trajectory based on distance from obstacles"""
        if not obstacles:
            return 1.0
            
        min_safe_distance = self.safety_params.get('safe_distance', 30)  # meters
        distances = []
        
        # Calculate minimum distance to each obstacle
        for obstacle in obstacles:
            for i in range(len(trajectory['latitude'])):
                point = {
                    'latitude': trajectory['latitude'][i],
                    'longitude': trajectory['longitude'][i],
                    'altitude': trajectory['altitude'][i]
                }
                distance = self._compute_distance(point, obstacle)
                distances.append(distance)
        
        min_distance = min(distances)
        if min_distance < min_safe_distance:
            return 0
        
        # Score decreases as we get closer to minimum safe distance
        safety_score = (min_distance - min_safe_distance) / min_safe_distance
        return min(1, safety_score)

    def _compute_weighted_score(self, scores):
        """Computes final weighted score based on individual criteria"""
        weights = {
            'length': 0.25,
            'smoothness': 0.25,
            'energy': 0.25,
            'safety': 0.25
        }
        
        return sum(scores[criterion] * weight 
                  for criterion, weight in weights.items() 
                  if criterion != 'total')

    def _compute_headings(self, trajectory):
        """Computes heading angles along trajectory"""
        headings = []
        for i in range(len(trajectory['latitude'])-1):
            dlat = trajectory['latitude'][i+1] - trajectory['latitude'][i]
            dlon = trajectory['longitude'][i+1] - trajectory['longitude'][i]
            heading = np.arctan2(dlon, dlat)
            headings.append(heading)
        return headings

    def _compute_distance(self, point1, point2):
        """Computes 3D distance between two points"""
        dlat = point2['latitude'] - point1['latitude']
        dlon = point2['longitude'] - point1['longitude']
        dalt = point2['altitude'] - point1['altitude']
        
        return sqrt(dlat**2 + dlon**2 + dalt**2)

    def _select_best_trajectory(self, scores):
        """Selects the best trajectory based on total scores"""
        return max(scores.keys(), key=lambda k: scores[k]['total'])
