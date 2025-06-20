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
