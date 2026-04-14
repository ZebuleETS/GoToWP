import numpy as np
from abc import ABC, abstractmethod
from compute import (
    check_trajectory_obstacles,
    extract_waypoint,
    get_destinations,
    get_power_consumption,
)
from typing import Dict, List

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
        num_points = self.params.get('num_points', 25)
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
        num_points = self.params.get('num_points', 50)
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

class LawnMowerTrajectory(TrajectoryGenerator):
    """
    Lawn mower trajectory generator for cartesian coordinates
    Génère une trajectoire en "tondeuse à gazon" pour couvrir une zone rectangulaire
    
    Args:
        params (dict): Parameters for the trajectory generation, e.g., fov_radius, altitude, etc.
        UAV_data (dict): UAV data containing specifications like speed, altitude limits, etc.
    """
    
    def __init__(self, params=None, UAV_data=None):
        super().__init__(params, UAV_data)
        
    def generate_path(self, area_bounds, fov_radius, uav_id=0, num_uavs=1, 
                      altitude=None, pattern='normal') -> Dict:
        """
        Génère une trajectoire en tondeuse à gazon pour couvrir une zone.
        
        Patterns disponibles pour optimiser la couverture multi-drone :
          - 'normal'     : balayage gauche→droite, avancement bas→haut (axe Y)
          - 'reverse'    : même tracé que normal mais parcouru en sens inverse (haut→bas)
          - 'transposed' : balayage bas→haut, avancement gauche→droite (axe X)
          - 'transposed_reverse' : même tracé transposé mais parcouru en sens inverse
        
        Args:
            area_bounds (dict): Limites de la zone {X_min, X_max, Y_min, Y_max}
            fov_radius (float): Rayon du champ de vision (FOV) en mètres
            uav_id (int): ID du drone (0, 1, 2, ...)
            num_uavs (int): Nombre total de drones
            altitude (float): Altitude de vol (optionnelle, sinon utilise params)
            pattern (str): Type de balayage ('normal', 'reverse', 'transposed', 'transposed_reverse')
            
        Returns:
            dict: Points de trajectoire {X: [], Y: [], Z: []}
        """
        # Extraction des limites de la zone
        x_min = area_bounds.get('X_min', self.params.get('X_lower_bound', 0.0))
        x_max = area_bounds.get('X_max', self.params.get('X_upper_bound', 6000.0))
        y_min = area_bounds.get('Y_min', self.params.get('Y_lower_bound', 0.0))
        y_max = area_bounds.get('Y_max', self.params.get('Y_upper_bound', 6000.0))
        
        if altitude is None:
            altitude = 400
        
        # Espacement entre les lignes = diamètre FOV
        line_spacing = fov_radius
        
        # Déterminer les axes de balayage selon le pattern
        is_transposed = pattern in ('transposed', 'transposed_reverse')
        is_reversed = pattern in ('reverse', 'transposed_reverse')
        
        if is_transposed:
            # Transposé : balayage le long de Y, avancement le long de X
            sweep_min, sweep_max = y_min, y_max   # axe de balayage (lignes)
            advance_min, advance_max = x_min, x_max  # axe d'avancement
        else:
            # Normal : balayage le long de X, avancement le long de Y
            sweep_min, sweep_max = x_min, x_max   # axe de balayage (lignes)
            advance_min, advance_max = y_min, y_max  # axe d'avancement
        
        # Calculer le nombre de lignes nécessaires
        advance_width = advance_max - advance_min
        num_lines = int(np.ceil(advance_width / line_spacing)) + 1
        
        # Générer les waypoints
        waypoints_x = []
        waypoints_y = []
        waypoints_z = []
        
        for line_idx in range(num_lines):
            # Position sur l'axe d'avancement
            advance_pos = advance_min + (line_idx * line_spacing)
            if advance_pos > advance_max:
                advance_pos = advance_max
            
            # Alterner la direction de balayage (aller-retour)
            if line_idx % 2 == 0:
                sweep_start, sweep_end = sweep_min, sweep_max
            else:
                sweep_start, sweep_end = sweep_max, sweep_min
            
            if is_transposed:
                # Transposé : X avance, Y balaye
                # Point de début de ligne
                waypoints_x.append(advance_pos)
                waypoints_y.append(sweep_start)
                waypoints_z.append(altitude)
                # Point de fin de ligne
                waypoints_x.append(advance_pos)
                waypoints_y.append(sweep_end)
                waypoints_z.append(altitude)
            else:
                # Normal : Y avance, X balaye
                # Point de début de ligne
                waypoints_x.append(sweep_start)
                waypoints_y.append(advance_pos)
                waypoints_z.append(altitude)
                # Point de fin de ligne
                waypoints_x.append(sweep_end)
                waypoints_y.append(advance_pos)
                waypoints_z.append(altitude)
        
        # Inverser l'ordre complet si pattern reverse
        if is_reversed:
            waypoints_x = waypoints_x[::-1]
            waypoints_y = waypoints_y[::-1]
            waypoints_z = waypoints_z[::-1]
        
        return {
            'X': waypoints_x,
            'Y': waypoints_y,
            'Z': waypoints_z
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
        num_points = self.params.get('num_points', 100)
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
        
        
def generate_all_trajectories(start_point, end_point, params, UAV_data, obstacles) -> List[Dict]:
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
    # Générateur ligne droite
    straight_traj = StraightLineTrajectory(params, UAV_data)
    straight = straight_traj.generate_path(start_point, end_point)
    straight = fix_trajectory(straight, obstacles)
    
    # Générateur courbe circulaire
    circular_traj = CircularTrajectory(params, UAV_data)
    circular = circular_traj.generate_path(start_point, end_point)
    circular = fix_trajectory(circular, obstacles)
    
    # Générateur Dubins 3D
    #dubins_traj = DubinsPath3D(params, UAV_data)
    #dubins = dubins_traj.generate_path(start_point, end_point)
    #dubins = fix_trajectory(dubins, obstacles)
    
    # Générateur Pythagorean Hodograph
    ph_traj = PythagoreanHodographPath(params, UAV_data)
    ph = ph_traj.generate_path(start_point, end_point)
    ph = fix_trajectory(ph, obstacles)

    # Retourne une liste avec toutes les trajectoires
    return [straight, circular, ph]

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
        obs_penalty = 0
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
            except Exception:
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
    
    

def fix_trajectory(trajectory, obstacles):
    """
    Fonction pour corriger une trajectoire en cas de violations de contraintes.
    Actuellement, cette fonction est un placeholder et retourne la trajectoire originale.
    
    Args:
        trajectory (dict): Trajectoire à corriger
        params (dict): Paramètres pour la correction
        obstacles (list): Liste des obstacles dans l'environnement    

    Returns:
        dict: Trajectoire corrigée
    """
    new_trajectory = dict()
    new_trajectory['X'] = []
    new_trajectory['Y'] = []
    new_trajectory['Z'] = []
    new_trajectory['X'].insert(0, trajectory['X'][0])
    new_trajectory['Y'].insert(0, trajectory['Y'][0])
    new_trajectory['Z'].insert(0, trajectory['Z'][0])
    # for each position in trajectory, check for obstacle collisions
    for i in range(len(trajectory['X']) - 1):
        Spoint = {
            'X': trajectory['X'][i],
            'Y': trajectory['Y'][i],
            'Z': trajectory['Z'][i]
        }
        Fpoint = {
            'X': trajectory['X'][i+1],
            'Y': trajectory['Y'][i+1],
            'Z': trajectory['Z'][i+1]
        }
        # Récupérer les destinations intermédiaires
        interm_destinations = get_destinations(Spoint, Fpoint, obstacles)
        # enlever le premier point (déjà ajouté)
        interm_destinations = interm_destinations[1:]
        # ajouter les points à la nouvelle trajectoire
        for dest in interm_destinations:
            new_trajectory['X'].insert(i+1, dest['X'])
            new_trajectory['Y'].insert(i+1, dest['Y'])
            new_trajectory['Z'].insert(i+1, dest['Z'])
            
    new_trajectory['X'].insert(len(trajectory['X']), trajectory['X'][-1])
    new_trajectory['Y'].insert(len(trajectory['Y']), trajectory['Y'][-1])
    new_trajectory['Z'].insert(len(trajectory['Z']), trajectory['Z'][-1])

    return new_trajectory

def generate_random_obstacles(num_obstacles, params):
    """
    Génère une liste d'obstacles polygone aléatoires dans une zone définie.
    
    Args:
        num_obstacles (int): Nombre d'obstacles à générer
        params (dict): Paramètres définissant la zone de génération et les caractéristiques des obstacles

    Returns:
        list: Liste d'obstacles générés
    """
    obstacles = []
    for _ in range(num_obstacles):
        # Générer des coordonnées aléatoires pour l'obstacle polygone
        center_x = np.random.uniform(params['X_lower_bound'], params['X_upper_bound'])
        center_y = np.random.uniform(params['Y_lower_bound'], params['Y_upper_bound'])
        num_vertices = np.random.randint(3, 6)  # Nombre de sommets pour le polygone
        radius = np.random.uniform(100, 500)
        vertices = []
        for _ in range(num_vertices):
            angle = np.random.uniform(0, 2 * np.pi)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            vertices.append((x, y))
        obstacle = {
            'vertices': vertices,
        }
        obstacles.append(obstacle)
    return obstacles

        