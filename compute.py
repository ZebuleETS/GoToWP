from math import atan, cos, sin, pi, sqrt, asin, atan2
import numpy as np

def get_power_consumption(flight_conditions):
    """
    Calculates UAV power consumption from strict MAVSDK telemetry.

    This function intentionally uses only the MAVSDK-based regression model
    (same as drone_core.predicted_power) and does not fallback to the legacy
    aerodynamic model.

    Args:
        flight_conditions (dict): Dictionary of current flight conditions.

    Returns:
        float: Power consumption (W).

    Raises:
        KeyError: If required MAVSDK telemetry keys are missing.
        ValueError: If telemetry values are None/non-numeric/non-finite.
    """
    required_keys = (
        'ground_speed_ms',
        'airspeed',
        'roll_rads',
        'pitch_rads',
        'yaw_rads',
        'relative_alt_m',
        'throttle_pct',
    )

    missing_keys = [k for k in required_keys if k not in flight_conditions]
    if missing_keys:
        raise KeyError(
            "Missing required MAVSDK telemetry fields for strict power model: "
            + ", ".join(missing_keys)
        )

    parsed = {}
    for key in required_keys:
        value = flight_conditions[key]
        if value is None:
            raise ValueError(f"MAVSDK telemetry field '{key}' is None")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"MAVSDK telemetry field '{key}' is not numeric: {value}")
        if not np.isfinite(value):
            raise ValueError(f"MAVSDK telemetry field '{key}' is not finite: {value}")
        parsed[key] = value

    gnd = parsed['ground_speed_ms']
    air = parsed['airspeed']
    roll = parsed['roll_rads']
    pitch = parsed['pitch_rads']
    yaw = parsed['yaw_rads']
    alt = parsed['relative_alt_m']
    thr = parsed['throttle_pct']

    # Handle both [0,1] and [0,100] throttle conventions.
    if 0.0 <= thr <= 1.0:
        thr *= 100.0

    predicted_power = (
        571.52 + (3.8 * gnd) - (14.85 * air) + (0.012 * roll)
        + (3.44 * pitch) - (0.058 * yaw) - (1.46 * alt) + (12.0 * thr)
    )
    return max(predicted_power, 0.0)


def get_shaft_power(uav_data, flight_conditions):
    """
    Calculates the shaft power required to provide the necessary thrust for the UAV.

    Args:
        uav_data (dict): Dictionary containing UAV parameters.
        flight_conditions (dict): Dictionary of current flight conditions.

    Returns:
        float: Shaft power (W).
    """
    thrust = compute_required_thrust(uav_data, flight_conditions)
    power = thrust * flight_conditions['airspeed']
    shaft_power = power / uav_data['propeller_efficiency']

    return shaft_power


def compute_required_thrust(uav_data, flight_conditions):
    """
    Calculates the required thrust to maintain flight under current conditions.

    Args:
        uav_data (dict): Dictionary containing UAV parameters.
        flight_conditions (dict): Dictionary of current flight conditions.

    Returns:
        float: Required thrust (N).
    """
    weight = uav_data['empty_weight'] + flight_conditions['weight']
    flight_path_angle = flight_conditions['flight_path_angle']
    grav_accel = flight_conditions['grav_accel']
    bank_angle = flight_conditions['bank_angle']
    airspeed_dot = flight_conditions['airspeed_dot']

    lift_to_drag = get_lift_to_drag(uav_data, flight_conditions)
    thrust = (cos(flight_path_angle) / (lift_to_drag * abs(cos(bank_angle))) +
              sin(flight_path_angle) + (airspeed_dot / grav_accel)) * weight * grav_accel

    return thrust


def get_lift_to_drag(uav_data, flight_conditions):
    """
    Calculates the lift-to-drag ratio (L/D) for the UAV under current flight conditions.

    Args:
        uav_data (dict): Dictionary containing UAV parameters.
        flight_conditions (dict): Dictionary of current flight conditions.

    Returns:
        float: Lift-to-drag ratio.
    """
    weight = uav_data['empty_weight'] + flight_conditions['weight']
    wing_area = uav_data['wing_area']
    wing_aspect_ratio = uav_data['wing_aspect_ratio']
    oswald_eff_ratio = uav_data['oswald_eff_ratio']
    zero_lift_drag = uav_data['zero_lift_drag']

    flight_path_angle = flight_conditions['flight_path_angle']
    grav_accel = flight_conditions['grav_accel']
    bank_angle = flight_conditions['bank_angle']
    airspeed = flight_conditions['airspeed']
    air_density = flight_conditions['air_density']

    lift = weight * grav_accel * cos(flight_path_angle) / abs(cos(bank_angle))
    lift_coef = 2 * lift / (air_density * airspeed ** 2 * wing_area)
    drag_coef = zero_lift_drag + (lift_coef ** 2 / (pi * wing_aspect_ratio * oswald_eff_ratio))

    lift_to_drag = lift_coef / drag_coef

    return lift_to_drag


def get_sink_rate(uav_data, flight_conditions):
    """
    Calculates the sink rate of the UAV in glide mode under current flight conditions.

    Args:
        uav_data (dict): Dictionary containing UAV parameters.
        flight_conditions (dict): Dictionary of current flight conditions.

    Returns:
        float: Sink rate (m/s).
    """
    weight = uav_data['empty_weight'] + flight_conditions['weight']
    wing_area = uav_data['wing_area']
    wing_aspect_ratio = uav_data['wing_aspect_ratio']
    oswald_eff_ratio = uav_data['oswald_eff_ratio']
    zero_lift_drag = uav_data['zero_lift_drag']

    grav_accel = flight_conditions['grav_accel']
    bank_angle = flight_conditions['bank_angle']
    airspeed = flight_conditions['airspeed']
    air_density = flight_conditions['air_density']
    airspeed_dot = flight_conditions['airspeed_dot']

    a = 2.0 * (weight * grav_accel) ** 2 / (air_density * airspeed ** 2
                                            * wing_area * pi * wing_aspect_ratio *
                                            oswald_eff_ratio * cos(bank_angle) ** 2)

    c = -(0.5 * air_density * airspeed ** 2 * wing_area * zero_lift_drag +
          (airspeed_dot * weight) + a)

    b = weight * grav_accel

    discr = b ** 2 - (4 * a * c)
    root_1 = (-b + sqrt(discr)) / (2 * a)
    root_2 = (-b - sqrt(discr)) / (2 * a)
    sin_gamma_a = max(root_1, root_2)

    sink_rate = sin_gamma_a * airspeed

    return sink_rate

def get_destination_from_range_and_bearing(starting_point, distance, bearing):
    """
    Calculates the geographic position reached after traveling a given distance along an initial bearing.

    Args:
        starting_point (dict): Starting point (latitude, longitude).
        distance (float): Distance to travel (m).
        bearing (float): Initial bearing (radians).

    Returns:
        tuple: (latitude, longitude) of the destination.
    """
    EARTH_RADIUS = 6378137
    lat = starting_point['latitude']
    lon = starting_point['longitude']

    angular_distance = distance / EARTH_RADIUS

    destination_lat = asin(sin(lat) * cos(angular_distance) +
                           cos(lat) * sin(angular_distance) * cos(bearing))

    destination_lon = lon + atan2(sin(bearing) * sin(angular_distance) * cos(lat), cos(angular_distance) -
                           sin(lat) * sin(destination_lat))

    if isinstance(destination_lon, np.floating):
        destination_lon = destination_lon.tolist()

    return destination_lat, destination_lon

def get_current_flight_data(FLT_track, FLT_conditions, nUAVs):
    """
    Extracts the current flight data for each UAV from the flight track and conditions.
    Args:
        FLT_track (list): List of dictionaries containing flight track data for each UAV.
        FLT_conditions (list): List of dictionaries containing flight conditions for each UAV.
        nUAVs (int): Number of UAVs.
    Returns:
        dict: Dictionary containing the latest flight data for each UAV.
    """
    FLT_data = {k: {} for k in range(nUAVs)}
    for u in range(nUAVs):
        FLT_data[u] = dict()
        FLT_data[u]['X'] = FLT_track[u]['X'][-1]
        FLT_data[u]['Y'] = FLT_track[u]['Y'][-1]
        FLT_data[u]['Z'] = FLT_track[u]['Z'][-1]
        FLT_data[u]['bearing'] = FLT_track[u]['bearing'][-1]
        FLT_data[u]['battery_capacity'] = FLT_track[u]['battery_capacity'][-1]
        FLT_data[u]['flight_mode'] = FLT_track[u]['flight_mode'][-1]
        FLT_data[u]['airspeed'] = FLT_conditions[u]['airspeed']
        FLT_data[u]['flight_path_angle'] = FLT_conditions[u]['flight_path_angle']
        FLT_data[u]['bank_angle'] = FLT_conditions[u]['bank_angle']
        FLT_data[u]['in_evaluation'] = FLT_track[u]['in_evaluation']
        FLT_data[u]['current_thermal_id'] = FLT_track[u]['current_thermal_id']
        FLT_data[u]['soaring_start_time'] = FLT_track[u]['soaring_start_time']

    return FLT_data

def geographic_to_cartesian(lat, lon, h=0):
    """
    Converts geographic coordinates (latitude, longitude, altitude) to 3D cartesian coordinates.

    Args:
        lat (float): Latitude in radians.
        lon (float): Longitude in radians.
        h (float, optional): Altitude in meters. Default is 0.

    Returns:
        tuple: (x, y, z) cartesian coordinates.
    """

    # LAT / LON SHOULD BE IN RADIANS

    EARTH_RADIUS = 6378137.0
    b = 6356752.314245
    e2 = 1 - (b / EARTH_RADIUS) ** 2
    n = EARTH_RADIUS / sqrt(1 - e2 * sin(lat) ** 2)

    x = (h + n) * cos(lat) * cos(lon)
    y = (h + n) * cos(lat) * sin(lon)
    z = (h + n - e2 * n) * sin(lat)

    return x, y, z

def cartesian_to_geographic(x, y, z):
    """
    Converts 3D cartesian coordinates to geographic coordinates (latitude, longitude, altitude).

    Args:
        x (float): x coordinate.
        y (float): y coordinate.
        z (float): z coordinate.

    Returns:
        tuple: (latitude, longitude, altitude).
    """
    # WGS84
    EARTH_RADIUS = 6378137.0
    b = 6356752.314245
    e2 = 1 - (b / EARTH_RADIUS) ** 2
    #e = sqrt(e2)

    lon = atan(y / x)
    #r = sqrt(x ** 2 + y ** 2 + z ** 2)
    p = sqrt(x ** 2 + y ** 2)

    lat_c = atan(p / z)
    lat = lat_c
    count = 0
    while (count < 6):
        Rn = EARTH_RADIUS / sqrt(1 - e2 * sin(lat) ** 2)
        h = p / cos(lat) - Rn
        tmp = 1.0 / (1 - e2 * (Rn / (Rn + h)))
        lat = atan((z / p) * tmp)
        count = count + 1

    return lat, lon, h

def compute_distance(pos, dest):
    """
    Calculates the 3D (horizontal and vertical) distance between two geographic points.

    Args:
        pos (dict): Start position (latitude, longitude, altitude).
        dest (dict): Destination position (latitude, longitude, altitude).

    Returns:
        list: List of distances (meters) for each destination.
    """
    if isinstance(pos['altitude'], list):
        pos_alt = pos['altitude'][-1]
    else:
        pos_alt = pos['altitude']

    dest_alt = dest['altitude']
    vert_dist = []

    if isinstance(dest_alt, float):
        dest_alt = [dest_alt]

    n = len(dest_alt)

    for i in range(n):
        if isinstance(dest_alt[i], list):
            dest_alt[i] = dest_alt[i][0]

        vert_dist.append(dest_alt[i] - pos_alt)

    horizontal_dis = compute_great_circle_dist(pos, dest)

    vert_dist_temp = [x ** 2 for x in vert_dist]
    horizontal_dis_temp = [x ** 2 for x in horizontal_dis]
    distance_temp = [sum(x) for x in zip(vert_dist_temp, horizontal_dis_temp)]
    distance = [sqrt(x) for x in distance_temp]
    return distance

def compute_great_circle_dist(pos, dest):
    """
    Calculates the great-circle distance between two points on the Earth's surface.

    Args:
        pos (dict): Start position (latitude, longitude).
        dest (dict): Destination position (latitude, longitude).

    Returns:
        list: Liste des distances (mètres) pour chaque destination.
    """
    if isinstance(pos['latitude'], list):
        pos_lat = pos['latitude'][-1]
        pos_lon = pos['longitude'][-1]
    else:
        pos_lat = pos['latitude']
        pos_lon = pos['longitude']

    dest_lat = dest['latitude']
    dest_lon = dest['longitude']

    if isinstance(dest_lat, float):
        dest_lat = [dest_lat]
        dest_lon = [dest_lon]

    n = len(dest_lat)

    EARTH_RADIUS = 6378137

    distance = []
    for i in range(n):
        if isinstance(dest_lat[i], list):
            dest_lat[i] = dest_lat[i][0]
            dest_lon[i] = dest_lon[i][0]

        dlon = dest_lon[i] - pos_lon
        dlat = dest_lat[i] - pos_lat

        a = sin(dlat / 2) ** 2 + cos(pos_lat) * cos(dest_lat[i]) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance.append(EARTH_RADIUS * c)

    return distance

def compute_bearing(pos, dest):
    """
    Calcule l'azimut initial entre deux points cartésiens.

    Args:
        pos (dict): Position de départ (X, Y).
        dest (dict): Position de destination (X, Y).

    Returns:
        list: Liste des azimuts (radians) pour chaque destination.
    """
    pos_lat = pos['latitude']
    pos_lon = pos['longitude']
    dest_lat = dest['latitude']
    dest_lon = dest['longitude']

    if isinstance(dest_lat, float):
        dest_lat = [dest_lat]
        dest_lon = [dest_lon]

    n = len(dest_lat)
    init_course = []
    for i in range(n):
        if isinstance(dest_lat[i], list):
            dest_lat[i] = dest_lat[i][0]
            dest_lon[i] = dest_lon[i][0]

        y = sin(dest_lon[i] - pos_lon) * cos(dest_lat[i])
        x = cos(pos_lat) * sin(dest_lat[i]) - sin(pos_lat) * cos(dest_lat[i]) * cos(dest_lon[i] - pos_lon)

        course = atan2(y, x)
        init_course.append(course + (2 * pi) % (2 * pi))
    return init_course

def extract_waypoint(waypoint_with_lists):
    """
    Convertit un waypoint au format liste en format scalaire
    
    Input: {'X': [val], 'Y': [val], 'Z': [val]}
    Output: {'X': val, 'Y': val, 'Z': val}
    """
    return {
        'X': waypoint_with_lists['X'][0],
        'Y': waypoint_with_lists['Y'][0],
        'Z': waypoint_with_lists['Z'][0]
    }
    
def compute_distance_cartesian(pos, dest):
    """
    Calcule la distance 3D euclidienne entre deux points cartésiens.

    Args:
        pos (dict): Position de départ (X, Y, Z).
        dest (dict): Position de destination (X, Y, Z).

    Returns:
        list: Liste des distances (mètres) pour chaque destination.
    """
    if isinstance(pos['Z'], list):
        pos_z = pos['Z'][-1]
    else:
        pos_z = pos['Z']

    dest_z = dest['Z']
    vert_dist = []

    if isinstance(dest_z, float):
        dest_z = [dest_z]

    n = len(dest_z)

    for i in range(n):
        if isinstance(dest_z[i], list):
            dest_z[i] = dest_z[i][0]

        vert_dist.append(dest_z[i] - pos_z)

    # Distance horizontale dans le plan X-Y
    horizontal_dis = compute_horizontal_distance_cartesian(pos, dest)

    # Distance euclidienne 3D
    vert_dist_temp = [x ** 2 for x in vert_dist]
    horizontal_dis_temp = [x ** 2 for x in horizontal_dis]
    distance_temp = [sum(x) for x in zip(vert_dist_temp, horizontal_dis_temp)]
    distance = [sqrt(x) for x in distance_temp]
    
    return distance

def compute_horizontal_distance_cartesian(pos, dest):
    """
    Calcule la distance horizontale (plan X-Y) entre deux points cartésiens.

    Args:
        pos (dict): Position de départ (X, Y).
        dest (dict): Position de destination (X, Y).

    Returns:
        list: Liste des distances (mètres) pour chaque destination.
    """
    if isinstance(pos['X'], list):
        pos_x = pos['X'][-1]
        pos_y = pos['Y'][-1]
    else:
        pos_x = pos['X']
        pos_y = pos['Y']

    dest_x = dest['X']
    dest_y = dest['Y']

    if isinstance(dest_x, float):
        dest_x = [dest_x]
        dest_y = [dest_y]

    n = len(dest_x)
    
    distance = []
    for i in range(n):
        if isinstance(dest_x[i], list):
            dest_x[i] = dest_x[i][0]
            dest_y[i] = dest_y[i][0]

        # Distance euclidienne dans le plan X-Y
        dx = dest_x[i] - pos_x
        dy = dest_y[i] - pos_y
        distance.append(sqrt(dx**2 + dy**2))

    return distance

def compute_bearing_cartesian(pos, dest):
    """
    Calcule l'azimut (bearing) entre deux points en coordonnées cartésiennes.
    
    Args:
        pos (dict): Position de départ (X, Y).
        dest (dict): Position de destination (X, Y).
    
    Returns:
        list: Liste des azimuts (radians) pour chaque destination.
    """
    if isinstance(pos['X'], list):
        pos_x = pos['X'][-1]
        pos_y = pos['Y'][-1]
    else:
        pos_x = pos['X']
        pos_y = pos['Y']

    dest_x = dest['X']
    dest_y = dest['Y']

    if isinstance(dest_x, float):
        dest_x = [dest_x]
        dest_y = [dest_y]

    n = len(dest_x)
    
    azimuts = []
    for i in range(n):
        if isinstance(dest_x[i], list):
            dest_x[i] = dest_x[i][0]
            dest_y[i] = dest_y[i][0]

        # Calculer les différences en X et Y
        dx = dest_x[i] - pos_x
        dy = dest_y[i] - pos_y
        
        # Calculer l'azimut dans le plan X-Y
        azimut = atan2(dy, dx)
        
        # Normaliser l'angle dans l'intervalle [0, 2π]
        azimut = (azimut + (2 * pi)) % (2 * pi)
        
        azimuts.append(azimut)
    
    return azimuts

def get_destination_from_range_and_bearing_cartesian(starting_point, distance, bearing):
    """
    Calcule la position cartésienne atteinte après avoir voyagé une distance donnée le long d'un azimut initial.

    Args:
        starting_point (dict): Point de départ (X, Y).
        distance (float): Distance à parcourir (m).
        bearing (float): Azimut initial (radians).

    Returns:
        tuple: (X, Y) de la destination.
    """
    x = starting_point['X']
    y = starting_point['Y']

    destination_x = x + distance * cos(bearing)
    destination_y = y + distance * sin(bearing)

    return destination_x, destination_y


def calculate_optimal_climb_angle(UAV_data, flight_conditions):
    """
    Calcule l'angle de montée optimal pour un drone à voilure fixe.
    
    Args:
        UAV_data (dict): Caractéristiques du drone
        flight_conditions (dict): Conditions de vol actuelles
    
    Returns:
        float: Angle de montée optimal en radians
    """
    
    # Copier les conditions de vol pour éviter de les modifier
    test_conditions = flight_conditions.copy()
    
    # Conditions pour vol horizontal
    test_conditions['flight_path_angle'] = 0.0
    test_conditions['bank_angle'] = 0.0
    
    # Calculer la puissance nécessaire pour vol horizontal
    level_power = get_power_consumption(test_conditions)
    
    # Calculer la puissance maximale disponible
    max_power = UAV_data['max_power_consumption'] * UAV_data['energy_conversion_efficiency']
    
    # Puissance excédentaire (différence entre puissance disponible et puissance requise)
    excess_power = max_power - level_power
    
    # Si pas de puissance excédentaire, pas de montée possible
    if excess_power <= 0:
        return 0.0
    
    # Calculer le poids
    weight = UAV_data['empty_weight'] * flight_conditions['grav_accel']
    
    # Calculer l'angle de montée théorique maximal basé sur la puissance excédentaire
    velocity = flight_conditions['airspeed']
    climb_ratio = excess_power / (weight * velocity)
    
    if climb_ratio >= 1.0:
        # Montée quasi-verticale théoriquement possible - contrainte aéro prendra le relais
        max_climb_angle_theory = np.deg2rad(85)  # Angle très élevé mais physiquement réaliste
    else:
        # Cas normal : calcul de l'angle via arcsin (ratio < 1)
        max_climb_angle_theory = np.arcsin(climb_ratio)

    max_allowed_angle = np.deg2rad(15)  # Limite pratique pour éviter le décrochage
    
    # Prendre le minimum entre l'angle théorique (puissance) et l'angle pratique (aéro)
    calculated_angle = min(max_climb_angle_theory, max_allowed_angle)
    
    return calculated_angle

def point_in_polygon(point, vertices):
    """
    Vérifie si un point est à l'intérieur d'un polygone défini par une liste de sommets.
    Args:
        point (tuple): Point à vérifier (x, y)
        vertices (list): Liste des sommets du polygone [(x1, y1), (x2, y2), ...]
    Returns:
        bool: True si le point est à l'intérieur du polygone, False sinon
    """
    x, y = point
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_point_in_obstacle(point, obstacle):
    """
    Vérifie si un point est à l'intérieur d'un obstacle polygonal.
    
    Args:
        point (dict): Point à vérifier {X, Y, Z}
        obstacle (dict): Obstacle polygonal {vertices: [(x1, y1), (x2, y2), ...]}
        
    Returns:
        bool: True si le point est dans l'obstacle, False sinon
    """
    # Vérifier la distance horizontale par rapport au centre de l'obstacle
    if 'X' not in obstacle or 'Y' not in obstacle:
        return False

    # Vérifier si le point est à l'intérieur des limites du polygone
    return point_in_polygon((point['X'], point['Y']), obstacle['vertices'])

def is_point_in_thermal(point, thermal):
    """
    Vérifie si un point est à l'intérieur d'un thermique cylindrique.
    
    Args:
        point (dict): Point à vérifier {X, Y, Z}
        thermal (dict): Obstacle cylindrique {X, Y, radius}
        
    Returns:
        bool: True si le point est dans l'obstacle, False sinon
    """
    # Vérifier si le point est à l'intérieur des limites de la thermal
    dx = point['X'] - thermal['X']
    dy = point['Y'] - thermal['Y']
    horizontal_distance = np.sqrt(dx**2 + dy**2)
    
    # Le point est dans le thermique si la distance horizontale est inférieure ou égale au rayon
    return horizontal_distance <= thermal['radius']

def check_segment_obstacle_collision(start_point, end_point, obstacle, num_checks=10):
    """
    Vérifie si un segment de ligne entre deux points traverse un obstacle.
    
    Args:
        start_point (dict): Point de départ {X, Y, Z}
        end_point (dict): Point d'arrivée {X, Y, Z}
        obstacle (dict): Obstacle cylindrique {x, y, radius, z_min, z_max}
        num_checks (int): Nombre de points intermédiaires à vérifier
        
    Returns:
        bool: True si le segment traverse l'obstacle, False sinon
    """
    # Vérifier les points de départ et d'arrivée
    if is_point_in_obstacle(start_point, obstacle) or is_point_in_obstacle(end_point, obstacle):
        return True
    
    # Générer des points intermédiaires le long du segment et vérifier chacun
    for i in range(1, num_checks):
        fraction = i / num_checks
        intermediate_point = {
            'X': start_point['X'] + fraction * (end_point['X'] - start_point['X']),
            'Y': start_point['Y'] + fraction * (end_point['Y'] - start_point['Y']),
            'Z': start_point['Z'] + fraction * (end_point['Z'] - start_point['Z'])
        }
        if is_point_in_obstacle(intermediate_point, obstacle):
            return True
    
    return False

def find_nearest_obstacle_distance(point, obstacles):
    """
    Calcule la distance au bord de l'obstacle le plus proche.
    
    Args:
        point (dict): Point à vérifier {X, Y, Z}
        obstacles (list): Liste des obstacles au format {'vertices': [...]}
        
    Returns:
        float: Distance au bord de l'obstacle le plus proche (négatif si à l'intérieur)
    """
    if not obstacles:
        return float('inf')
    
    min_distance = float('inf')
    
    for obstacle in obstacles:
        if 'vertices' not in obstacle:
            continue
            
        vertices = np.array(obstacle['vertices'])
        
        # Calculer la distance minimale à chaque segment du polygone
        min_dist_to_obstacle = float('inf')
        
        for i in range(len(vertices)):
            # Points du segment
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            
            # Point à vérifier
            px = point['X']
            py = point['Y']
            
            # Vecteur du segment
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Longueur du segment au carré
            segment_length_sq = dx**2 + dy**2
            
            if segment_length_sq == 0:
                # Le segment est un point
                dist = np.sqrt((px - p1[0])**2 + (py - p1[1])**2)
            else:
                # Projection du point sur la ligne du segment
                t = max(0, min(1, ((px - p1[0]) * dx + (py - p1[1]) * dy) / segment_length_sq))
                
                # Point le plus proche sur le segment
                closest_x = p1[0] + t * dx
                closest_y = p1[1] + t * dy
                
                # Distance au point le plus proche
                dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
            
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)
        
        # Vérifier si le point est à l'intérieur du polygone
        if point_in_polygon((px, py), obstacle['vertices']):
            # Si à l'intérieur, la distance est négative
            min_dist_to_obstacle = -min_dist_to_obstacle
        
        # Mettre à jour la distance minimale globale
        if abs(min_dist_to_obstacle) < abs(min_distance):
            min_distance = min_dist_to_obstacle
    
    return min_distance

def find_nearest_waypoint(current_pos, GOAL_WPs, obstacles, exit_thermal, wp_idx):
    """
    Finds the index of the nearest waypoint to the current position.

    Args:
        GOAL_WPs (dict): Dictionary containing goal waypoints with 'X' and 'Y' keys.
        current_pos (dict): Current position with 'X' and 'Y' keys.

    Returns:
        int: Index of the nearest waypoint.
    """
    total_wps = len(GOAL_WPs.get('X', []))
    if total_wps == 0:
        return 0

    # Borner l'index de départ pour éviter les bornes invalides
    start_idx = max(0, min(wp_idx, total_wps - 1))

    # Calculer toutes les distances en une seule fois
    distances = compute_horizontal_distance_cartesian(current_pos, GOAL_WPs)
    
    # Filtrer les waypoints qui sont en collision avec des obstacles
    valid_indices = []
    for i in range(start_idx, total_wps):
        waypoint = {
            'X': GOAL_WPs['X'][i],
            'Y': GOAL_WPs['Y'][i],
            'Z': GOAL_WPs['Z'][i]
        }
        is_valid = True
        for obstacle in obstacles:
            if is_point_in_obstacle(waypoint, obstacle):
                is_valid = False
                break
        
        # Vérifier si le waypoint est à l'intérieur du thermique de sortie
        if exit_thermal is not None:
            thermal = {
                'X': exit_thermal.x,
                'Y': exit_thermal.y,
                'radius': exit_thermal.radius
            }
            if is_point_in_thermal(waypoint, thermal):
                is_valid = False
        
        if is_valid:
            valid_indices.append(i)

    # Fallback robuste : si tous les points sont filtrés, choisir le plus proche
    # parmi les waypoints restants (et préférer hors thermique si possible).
    if not valid_indices:
        fallback_indices = list(range(start_idx, total_wps))

        if exit_thermal is not None:
            non_thermal_indices = []
            thermal = {
                'X': exit_thermal.x,
                'Y': exit_thermal.y,
                'radius': exit_thermal.radius
            }
            for i in fallback_indices:
                waypoint = {
                    'X': GOAL_WPs['X'][i],
                    'Y': GOAL_WPs['Y'][i],
                    'Z': GOAL_WPs['Z'][i]
                }
                if not is_point_in_thermal(waypoint, thermal):
                    non_thermal_indices.append(i)

            if non_thermal_indices:
                fallback_indices = non_thermal_indices

        valid_indices = fallback_indices

    nearest_index = min(valid_indices, key=lambda i: distances[i])
    
    return nearest_index

def check_trajectory_obstacles(trajectory, obstacles):
    """
    Vérifie si une trajectoire entre en collision avec des obstacles.
    
    Args:
        trajectory (dict): Trajectoire {X: [], Y: [], Z: []}
        obstacles (list): Liste des obstacles
        
    Returns:
        tuple: (collision_exists, collision_points, min_distance)
            - collision_exists (bool): True si une collision existe
            - collision_points (list): Indices des points en collision
            - min_distance (float): Distance minimale aux obstacles
    """
    collision_exists = False
    collision_points = []
    min_distance = float('inf')
    
    if not obstacles:
        return False, [], min_distance
    
    # Vérifier chaque point de la trajectoire
    for i in range(len(trajectory['X'])):
        point = {
            'X': trajectory['X'][i],
            'Y': trajectory['Y'][i],
            'Z': trajectory['Z'][i]
        }
        
        for obstacle in obstacles:
            if is_point_in_obstacle(point, obstacle):
                collision_exists = True
                collision_points.append(i)
                break
                
        # Calculer la distance minimale aux obstacles
        distance = find_nearest_obstacle_distance(point, obstacles)
        min_distance = min(min_distance, distance)
    
    # Vérifier également les segments entre les points
    for i in range(len(trajectory['X']) - 1):
        start_point = {
            'X': trajectory['X'][i],
            'Y': trajectory['Y'][i],
            'Z': trajectory['Z'][i]
        }
        end_point = {
            'X': trajectory['X'][i + 1],
            'Y': trajectory['Y'][i + 1],
            'Z': trajectory['Z'][i + 1]
        }
        
        for obstacle in obstacles:
            if check_segment_obstacle_collision(start_point, end_point, obstacle):
                collision_exists = True
                collision_points.append(i)
                break
    
    return collision_exists, collision_points, min_distance

def wrapN(x, n):
    return np.mod(x, n)

def computePolyArea(x,y):
    A = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return A

def sum_normalizedMatrixColumns(M):
    m, n = M.shape
    s = np.array([0.0] * m)
    for i in range(n):
        c = M[:, i]
        norm_c = np.divide(c, np.linalg.norm(c))
        norm_c[np.isnan(norm_c)] = 0
        s += norm_c

    return s

def getDomination(x, y):
    x = np.array(x)
    y = np.array(y)
    return all(x <= y) and any(x < y)


def ParetRanking(costs):
    n = costs.shape[0]
    is_dominated = np.zeros(n, dtype = int)
    ranks = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(n):
            if j != i:
                flag = getDomination(costs[i, :], costs[j, :])
                is_dominated[i] += int(not flag)

    fronts = np.unique(is_dominated).tolist()
    for k in range(len(fronts)):
        ranks[is_dominated == fronts[k]] = k + 1

    return ranks


def decision_making(DM):
    """
    Effectue la prise de décision multi-critères en utilisant le classement de Pareto.
    
    Args:
        DM (numpy.ndarray): Matrice de décision où chaque ligne représente une alternative
                           et chaque colonne un critère.
    
    Returns:
        numpy.ndarray: Indices des alternatives classées de la meilleure à la pire.
    """
     # Vérifier si la matrice est vide
    if DM.shape[0] == 0:
        return np.array([])
    
    # Vérifier si c'est une seule alternative
    if DM.shape[0] == 1:
        return np.array([0])
    
    # Vérifier s'il y a des valeurs à traiter
    if DM.shape[1] == 0:
        return np.arange(DM.shape[0])
    
    
    D1 = -np.sum(abs(DM - np.mean(DM, axis=0)), axis=1).reshape(-1, 1)
    D2 = np.sum(abs(DM - np.min(DM, axis=0)), axis=1).reshape(-1, 1)
    m, n = DM.shape
    D3 = []
    D4 = []
    for i in range(m):
        temp = np.vstack((np.arange(n), DM[i,:]))
        P = 0
        for j in range(n):
            P += np.linalg.norm(temp[:, j] - temp[:, wrapN(j + 1, n)])
        A = sqrt(computePolyArea(temp[0,:], temp[1,:]))
        D3.append(P)
        D4.append(A)
    D3 = np.array(D3).reshape(-1, 1)
    D4 = np.array(D4).reshape(-1, 1)

    D5 = np.zeros(DM.shape[0], dtype=int)
    for j in range(n):
        r = ParetRanking(DM[:, j].reshape(-1, 1))
        D5 += + r
    D5 = D5.reshape(-1, 1)

    D6 = sum_normalizedMatrixColumns(DM)
    D6 = D6

    newDM = np.hstack((D1, D2, D3, D4, D5))
    ranks = ParetRanking(newDM)
    MaxFNo = np.max(ranks)
    DT = np.arange(DM.shape[0])
    ranked_DT = []

    for i in range(MaxFNo):
        temp = DT[ranks == i+1]
        sorted_indices = np.argsort(D6[temp])
        ranked_DT.append(temp[sorted_indices])

    # Concaténer tous les rangs dans l'ordre au lieu d'essayer de créer un array 2D
    if len(ranked_DT) > 0:
        all_ranked_indices = np.concatenate(ranked_DT)
    else:
        all_ranked_indices = np.arange(DM.shape[0])

    return all_ranked_indices # indices de toutes les options du meillieure jusqu'au pire

def get_climb_rate(thermal_velocity, soaring_time, params):

    # note: utiliser l'altitude courante
    altitude = thermal_velocity * soaring_time

    minimum_altitude_msl = params['Z_lower_bound']
    maximum_altitude_msl = params['Z_upper_bound']
    working_floor = params['working_floor']
    slope = 0.0008
    min_acceptable_climb = 0.3

    alt_clearance = altitude - minimum_altitude_msl
    alt_range = maximum_altitude_msl - minimum_altitude_msl

    if alt_clearance < 0.1 * alt_range:
        intercept = 0
        slope = min_acceptable_climb / (0.1 * alt_range)
        offset = 0
    elif altitude < working_floor:
        intercept = min_acceptable_climb
        slope = 0.0
        offset = 0.0

    elif alt_clearance > 0.9 * alt_range:
        intercept = min_acceptable_climb + min(0.8 * alt_range, (maximum_altitude_msl - working_floor)) * slope;
        slope = 10.0 * slope
        offset = 0.9 * alt_range

    else:
        intercept = min_acceptable_climb
        offset = max(0.1 * alt_range, (working_floor - minimum_altitude_msl))


    climb_rate = max(slope * (alt_clearance - offset) + intercept, 0.1)

    return climb_rate

def turn_radius(V, phi_rad):
    """
    Calcule le rayon de virage pour une vitesse et un angle d'inclinaison donnés.
    
    Args:
        V (float): Vitesse de vol [m/s]
        phi_rad (float): Angle d'inclinaison [radians]
    
    Returns:
        float: Rayon de virage [m]
    """
    g = 9.81  # gravité [m/s²]
    return V**2 / (g * np.tan(phi_rad))

def optimal_radius(thermal, UAV_data, flight_conditions):
    """
    Calcule le rayon optimal de virage dans un thermique pour maximiser le taux de montée net.
    
    Args:
        thermal (Thermal): Objet thermique
        UAV_data (dict): Données du UAV
        flight_conditions (dict): Conditions de vol actuelles
        
    Returns:
        tuple: (rayon_optimal, vitesse_optimale, angle_inclinaison_optimal_rad)
            - rayon_optimal (float): Rayon de virage optimal [m]
            - vitesse_optimale (float): Vitesse optimale [m/s]
            - angle_inclinaison_optimal_rad (float): Angle d'inclinaison optimal [radians]
    """
    # Plages d'angles d'inclinaison et de vitesses à tester
    bank_angles_rad = np.linspace(np.radians(15), np.radians(45), 50)  # Convertir en radians
    airspeeds = np.linspace(UAV_data['min_airspeed'], UAV_data['max_airspeed'], 30)
    
    best_net_climb = -999
    best_radius = None
    best_bank_angle_rad = None
    best_airspeed = None
    
    # Copier les conditions de vol pour les tests
    test_conditions = flight_conditions.copy()
    
    for airspeed in airspeeds:
        for bank_angle_rad in bank_angles_rad:
            # Calculer le rayon de virage pour cette vitesse et cet angle
            radius = turn_radius(airspeed, bank_angle_rad)
            
            # PX4 do_orbit nécessite un rayon minimum pour orbiter proprement
            if radius < 25.0:
                continue
            
            # Mettre à jour les conditions de vol pour le calcul du sink rate
            test_conditions['airspeed'] = airspeed
            test_conditions['bank_angle'] = bank_angle_rad  # En radians
            test_conditions['flight_path_angle'] = 0.0  # Vol en palier dans le virage
            
            # Calculer le taux de descente en virage (sink rate)
            sink_rate = get_sink_rate(UAV_data, test_conditions)
            
            # Force thermique moyenne sur le cercle de virage
            avg_thermal_strength = 0
            n_points = 20
            for i in range(n_points):
                angle = 2 * pi * i / n_points
                x_circle = thermal.x + radius * cos(angle)
                y_circle = thermal.y + radius * sin(angle)
                dist = sqrt((x_circle - thermal.x)**2 + (y_circle - thermal.y)**2)
                avg_thermal_strength += thermal.get_lift_rate(dist)

            avg_thermal_strength /= n_points

            net_climb_rate = avg_thermal_strength - sink_rate

            # Vérifier si cette combinaison est meilleure
            if net_climb_rate > best_net_climb:
                best_net_climb = net_climb_rate
                best_radius = radius
                best_bank_angle_rad = bank_angle_rad
                best_airspeed = airspeed
    
    # Si aucune solution viable n'est trouvée, utiliser des valeurs par défaut
    if best_radius is None:
        best_radius = 50.0  # Rayon par défaut de 50m
        best_airspeed = 8.0
        best_bank_angle_rad = np.radians(30.0)  # 30° converti en radians
    
    return best_radius, best_airspeed, best_bank_angle_rad

def calculate_optimal_soaring_parameters(UAV_data, thermal, flight_conditions):
    """
    Calcule les paramètres optimaux de soaring pour un UAV dans un thermique donné.
    
    Args:
        UAV_data (dict): Données du UAV
        thermal (Thermal): Objet thermique
        flight_conditions (dict): Conditions de vol actuelles
    
    Returns:
        dict: Paramètres optimaux de soaring (tous les angles en radians)
    """
    # Calcul du rayon optimal
    optimal_r, optimal_speed, optimal_bank_angle_rad = optimal_radius(thermal, UAV_data, flight_conditions)

    return {
        'optimal_radius': optimal_r,
        'optimal_speed': optimal_speed,
        'optimal_bank_angle': optimal_bank_angle_rad,  # En radians
    }
    
# Fix trajectory collision functions
def EdgeCollision(Edge1, Edge2):
    tol = 0.0
    eps = 2.2204e-16
    p1 = Edge1[0:2]
    p2 = Edge1[2:4]
    p3 = Edge2[0:2]
    p4 = Edge2[2:4]

    colliding = 0
    if (abs((p1[0] - p3[0]) <= tol) and (abs(p1[1] - p3[1]) <= tol) and (abs(p2[0] - p4[0]) <= tol) and (abs(p2[1] - p4[1]) <= tol)):
        colliding = 1
        return colliding

    temp1 = p1 - p3
    temp2 = p2 - p3
    temp3 = p3 - p1
    temp4 = p4 - p1
    temp5 = p4 - p3
    temp6 = p2 - p1
    M1 = np.array([[temp1[0], temp2[0], temp3[0], temp4[0]], [temp1[1], temp2[1], temp3[1], temp4[1]], [0, 0, 0, 0]])
    M2 = np.array([[temp5[0], temp5[0], temp6[0], temp6[0]], [temp5[1], temp5[1], temp6[1], temp6[1]], [0, 0, 0, 0]])

    d = np.array([]).reshape(3, 0)
    for i in range(4):
        temp = np.cross(M1[:,i], M2[:,i]).reshape(3,1)
        d = np.concatenate((d, temp), axis=1)
    d = d[-1,:]

    if ((d[0] > 0 and d[1] < 0) or (d[0] < 0 and d[1] > 0)) and ((d[2] > 0 and d[3] < 0) or (d[2] < 0 and d[3] > 0)):
        colliding = 1
    elif(abs(d[0]) < 100 * eps and OnSegment(p3, p4, p1)):
        colliding = 1
    elif(abs(d[1]) < 100 * eps and OnSegment(p3, p4, p2)):
        colliding = 1
    elif(abs(d[2]) < 100 * eps and OnSegment(p1, p2, p3)):
        colliding = 1
    elif(abs(d[3]) < 100 * eps and OnSegment(p1, p2, p4)):
        colliding = 1

    if colliding == 1:
        A1, b1 = EdgePtsToVec(Edge1)
        A2, b2 = EdgePtsToVec(Edge2)
        colliding = AffineIntersect(np.array([[A1[0], A1[1]], [A2[0], A2[1]]]), np.concatenate((b1, b2), axis=0))

    return colliding

def OnSegment(pi, pj, pk):
    val = 0
    if (min(pi[0], pj[0]) <= pk[0] and pk[0] <= max(pi[0], pj[0])) and (min(pi[1], pj[1]) <= pk[1] and pk[1] <= max(pi[1], pj[1])):
        val = 1
    return val

def AffineIntersect(A, b):
    for i in range(2):
        for j in range(2):
            if np.isnan(A[i,j]) or np.isinf(A[i,j]) or A[i,j] > 1e6:
                A[i,j] = 1e6

    if (np.linalg.cond(A) > 1e6):
        doesIntersect = 0
        return doesIntersect

    doesIntersect = 1
    return doesIntersect

def EdgePtsToVec(edge):
    f = np.array([edge[0], edge[1], 0])
    g = np.array([edge[2], edge[3], 0])
    A = np.cross(f - g, np.array([0, 0, 1]))
    A = A / np.linalg.norm(A)
    b = np.multiply(A, f)
    return A[0:2], b

def CheckCollision(ptA, ptB, obstEdges):
    n = obstEdges.shape[0]
    inCollision = 0
    for k in range(n):
        cnd1 = (max(ptA[0], ptB[0]) < min(obstEdges[k, 0], obstEdges[k, 2]))
        cnd2 = (min(ptA[0], ptB[0]) > max(obstEdges[k, 0], obstEdges[k, 2]))
        cnd3 = (max(ptA[1], ptB[1]) < min(obstEdges[k, 1], obstEdges[k, 3]))
        cnd4 = (min(ptA[1], ptB[1]) > max(obstEdges[k, 1], obstEdges[k, 3]))
        cnd = cnd1 or cnd2 or cnd3 or cnd4
        if not cnd:
            cnd5 = EdgeCollision(np.array([ptA[0], ptA[1], ptB[0], ptB[1]]), obstEdges[k,:])
            if cnd5:
                if (sum(abs(ptA - obstEdges[k, 0:2])) > 0 and
                        sum(abs(ptB-obstEdges[k, 0:2])) > 0 and
                        sum(abs(ptA - obstEdges[k, 2:4])) > 0 and
                        sum(abs(ptB - obstEdges[k, 2:4])) > 0):
                    inCollision = 1
                    return inCollision
    return inCollision

def get_destinations(startPos, endPos, obstacles):
    """
    Calcule un chemin entre deux points en évitant les obstacles polygonaux.
    Utilise un graphe de visibilité + Dijkstra pour trouver le chemin le plus court.
    
    Args:
        startPos (dict): Point de départ {X, Y, Z}
        endPos (dict): Point d'arrivée {X, Y, Z}
        obstacles (list): Liste d'obstacles au format {'vertices': np.array([(x1,y1), (x2,y2), ...])}
        
    Returns:
        list: Liste de points formant le chemin [{X, Y, Z}, ...]
    """
    import numpy as np
    import heapq
    
    # Convertir les obstacles en format numpy array
    obstacle_arrays = []
    for obs in obstacles:
        if isinstance(obs, dict) and 'vertices' in obs:
            vertices = np.array(obs['vertices']) if not isinstance(obs['vertices'], np.ndarray) else obs['vertices']
            obstacle_arrays.append(vertices)
        elif isinstance(obs, np.ndarray):
            obstacle_arrays.append(obs)
    
    # Si pas d'obstacles, retourner une ligne droite
    if len(obstacle_arrays) == 0:
        return [startPos, endPos]
    
    numObsts = len(obstacle_arrays)
    allGraphPts = np.array([]).reshape(0, 2)
    obsEdges = np.array([]).reshape(0, 4)
    
    for i in range(numObsts):
        numOPts = obstacle_arrays[i].shape[0]
        allGraphPts = np.concatenate((allGraphPts, obstacle_arrays[i]), axis=0)
        obsEdges = np.concatenate((obsEdges, np.concatenate(
            (obstacle_arrays[i], obstacle_arrays[i][np.mod(np.arange(numOPts) + 1, numOPts), :]), axis=1)), axis=0)

    # Convertir startPos et endPos en format numpy array 2D
    startPosArray = np.array([[startPos['X'], startPos['Y']]])
    endPosArray = np.array([[endPos['X'], endPos['Y']]])
    
    allGraphPts = np.concatenate((allGraphPts, startPosArray), axis=0)
    allGraphPts = np.concatenate((allGraphPts, endPosArray), axis=0)

    n = allGraphPts.shape[0]
    id_source = n - 2  # avant-dernier = start
    id_target = n - 1  # dernier = end

    # Construire la matrice d'adjacence (visibilité)
    A = np.ones((n, n)) - np.eye(n)
    ptCount = -1
    
    for i in range(numObsts):
        numOPts = obstacle_arrays[i].shape[0]
        X = np.arange(ptCount + 1, ptCount + numOPts + 1)
        for j in range(len(X)):
            A[X, X[j]] = 0
        for k in range(numOPts - 1):
            A[ptCount + 1 + k, ptCount + k + 2] = 1
            A[ptCount + k + 2, ptCount + 1 + k] = 1
        A[ptCount + 1, ptCount + numOPts] = 1
        A[ptCount + numOPts, ptCount + 1] = 1
        ptCount += numOPts

    for i in range(n):
        for j in range(i, n):
            inColl = CheckCollision(allGraphPts[i, :], allGraphPts[j, :], obsEdges)
            if inColl == 1:
                A[i, j] = 0
                A[j, i] = 0

    # --- Dijkstra sur le graphe de visibilité ---
    dist = np.full(n, np.inf)
    dist[id_source] = 0.0
    prev = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    
    # File de priorité : (distance, node_id)
    heap = [(0.0, id_source)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if u == id_target:
            break
        for v in range(n):
            if A[u, v] > 0 and not visited[v]:
                edge_len = np.linalg.norm(allGraphPts[u] - allGraphPts[v])
                new_dist = d + edge_len
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))
    
    # Reconstituer le chemin
    if dist[id_target] == np.inf:
        # Pas de chemin trouvé → ligne droite (fallback)
        return [startPos, endPos]
    
    path_indices = []
    node = id_target
    while node != -1:
        path_indices.append(node)
        node = prev[node]
    path_indices.reverse()
    
    # Interpoler l'altitude linéairement le long du chemin
    total_len = 0.0
    seg_lengths = []
    for k in range(1, len(path_indices)):
        sl = np.linalg.norm(allGraphPts[path_indices[k]] - allGraphPts[path_indices[k - 1]])
        seg_lengths.append(sl)
        total_len += sl
    
    result = []
    cumulative = 0.0
    for k, idx in enumerate(path_indices):
        if total_len > 0:
            frac = cumulative / total_len
        else:
            frac = 0.0
        z = startPos['Z'] + frac * (endPos['Z'] - startPos['Z'])
        result.append({
            'X': float(allGraphPts[idx, 0]),
            'Y': float(allGraphPts[idx, 1]),
            'Z': float(z),
        })
        if k < len(seg_lengths):
            cumulative += seg_lengths[k]
    
    return result

def convert_cylindrical_obstacles_to_polygons(obstacles, num_points=16):
    """
    Convertit des obstacles cylindriques en polygones approximatifs.
    
    Args:
        obstacles (list): Liste des obstacles cylindriques avec 'X', 'Y', 'radius'
        num_points (int): Nombre de points pour approximer le cercle
        
    Returns:
        list: Liste de tableaux numpy contenant les sommets des polygones
    """
    polygon_obstacles = []
    
    for obs in obstacles:
        # Générer des points autour du cercle
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        vertices = np.zeros((num_points, 2))
        
        for i, angle in enumerate(angles):
            vertices[i, 0] = obs['X'] + obs['radius'] * np.cos(angle)
            vertices[i, 1] = obs['Y'] + obs['radius'] * np.sin(angle)
        
        polygon_obstacles.append(vertices)
    
    return polygon_obstacles