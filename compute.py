from math import atan, cos, sin, pi, sqrt, asin, atan2
import numpy as np

def get_power_consumption(uav_data, flight_conditions):
    """
    Calculates the total power consumption of the UAV considering energy conversion efficiency.

    Args:
        uav_data (dict): Dictionary containing UAV parameters.
        flight_conditions (dict): Dictionary of current flight conditions.

    Returns:
        float: Power consumption (W).
    """
    power_consumption = (get_shaft_power(uav_data, flight_conditions) /
                         uav_data['energy_conversion_efficiency'])

    return power_consumption


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
