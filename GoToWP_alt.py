from math import cos, sin, pi, sqrt, atan2
from typing import Dict, List, Tuple
import numpy as np
import warnings
from compute import (
    compute_great_circle_dist,
    get_current_flight_data,
    get_power_consumption,
    get_sink_rate,
    get_destination_from_range_and_bearing
)
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#not used
def update_remaining_energy(flight_conditions, power_consumption, time_step):
    """
    Updates the remaining battery energy after a given flight duration.

    Args:
        flight_conditions (dict): Dictionary of current flight conditions.
        power_consumption (float): Power consumption (W).
        time_step (float or list): Flight duration(s) (s).

    Returns:
        list: List of remaining energy (Wh) after each time step.
    """
    remaining_energy = []

    if isinstance(time_step, float):
        time_step = [time_step]

    n = len(time_step)

    for i in range(n):
        energy_cost = power_consumption * (time_step[i] / 3600)
        remaining_energy.append(flight_conditions['battery_capacity'] - energy_cost)

    return remaining_energy


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
        list: List of distances (meters) for each destination.
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

#not used
def compute_bearing(pos, dest):
    """
    Calculates the initial bearing (azimuth) between two geographic points.

    Args:
        pos (dict): Start position (latitude, longitude).
        dest (dict): Destination position (latitude, longitude).

    Returns:
        list: List of bearings (radians) for each destination.
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


def find_min_index(list):
    """
    Returns the index of the minimum value in a list.

    Args:
        list (list): List of numeric values.

    Returns:
        int: Index of the minimum value.
    """
    minimum_index = min(range(len(list)), key=lambda i: list[i])
    return minimum_index

def lineXline(pA, pB):
    """
    Calculates the intersection point between two lines in 3D space.

    Args:
        pA (np.ndarray): Points defining the first line (2x3).
        pB (np.ndarray): Points defining the second line (2x3).

    Returns:
        np.ndarray: Coordinates of the intersection point (3x1).
    """

    SI = pA - pB
    NI = np.divide(SI, np.sqrt(np.sum(np.power(SI, 2), axis=0)))
    nX = NI[:, 0]
    nY = NI[:, 1]
    nZ = NI[:, 2]
    SXX = np.sum(np.power(nX, 2) - 1)
    SYY = np.sum(np.power(nY, 2) - 1)
    SZZ = np.sum(np.power(nZ, 2) - 1)
    SXY = np.sum(np.multiply(nX, nY))
    SXZ = np.sum(np.multiply(nX, nZ))
    SYZ = np.sum(np.multiply(nY, nZ))
    S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])
    CX = np.sum(np.multiply(pA[:, 0], np.power(nX, 2) - 1) + np.multiply(pA[:, 1], np.multiply(nX, nY)) + np.multiply(pA[:, 2], np.multiply(nX, nZ)))
    CY = np.sum(np.multiply(pA[:, 0], np.multiply(nX, nY)) + np.multiply(pA[:, 1], np.power(nY, 2) - 1) + np.multiply(pA[:, 2], np.multiply(nY, nZ)))
    CZ = np.sum(np.multiply(pA[:, 0], np.multiply(nX, nZ)) + np.multiply(pA[:, 1], np.multiply(nY, nZ)) + np.multiply(pA[:, 2], np.power(nZ, 2) - 1))
    C = np.array([[CX], [CY], [CZ]])
    P_intersect = np.linalg.solve(S, C)

    return P_intersect

@dataclass
class Position:
    """Class to represent a geographic position"""
    latitude: float
    longitude: float
    altitude: float

@dataclass
class TrajectoryCandidate:
    """Class to represent a trajectory candidate"""
    position: Position
    bearing: float
    battery_capacity: float
    flight_mode: str
    airspeed: float
    flight_path_angle: float

class TrajectoryOptimizer:
    """Class to optimize UAV trajectories"""
    
    def __init__(self, params: Dict, uav_data: Dict):
        self.params = params
        self.uav_data = uav_data
        self._extract_parameters()
    
    def _extract_parameters(self):
        """Extract and store frequently used parameters"""
        self.altitude_bounds = (
            self.params['altitude_lower_bound'],
            self.params['altitude_upper_bound']
        )
        self.time_step = self.params['time_step']
        self.bearing_step = self.params['bearing_step']
        self.speed_step = self.params['speed_step']
        self.safe_distance = self.params['safe_distance']
        self.horizon_length = self.params['horizon_length']
        self.waypoint_threshold = self.params.get('waypoint_threshold', 10.0)
        
        self.max_turn_rate = self.uav_data['max_turn_rate']
        self.velocity_bounds = (
            self.uav_data['min_airspeed'],
            self.uav_data['max_airspeed']
        )

def gotoWaypoint(FLT_track: Dict, FLT_conditions: Dict, GOAL_WPs: Dict, 
                 nUAVs: int, Uidx: int, params: Dict, UAV_data: Dict, 
                 current_wp_idx: int) -> Tuple[Dict, Dict, int]:
    """
    Generates and selects the best trajectory for a given UAV considering physical, energy, and safety constraints,
    then updates its position and flight conditions. Uses TrajectoryEvaluator to select the optimal path.

    Args:
        FLT_track (dict): History of UAV positions and states.
        FLT_conditions (dict): Current flight conditions of UAVs.
        GOAL_WPs (dict): Target waypoints (latitude, longitude).
        nUAVs (int): Total number of UAVs.
        Uidx (int): Index of the UAV to process.
        params (dict): Simulation parameters.
        UAV_data (dict): Physical parameters of the UAV.
        current_wp_idx (int): Index of the current waypoint to follow in GOAL_WPs.

    Returns:
        tuple: (FLT_track, FLT_conditions, current_wp_idx) updated for the processed UAV.
    """
    
    # Preliminary checks
    if current_wp_idx >= len(GOAL_WPs['latitude']):
        return FLT_track, FLT_conditions, current_wp_idx
    
    optimizer = TrajectoryOptimizer(params, UAV_data)
    
    # Current position and target waypoint
    current_pos = Position(
        FLT_track[Uidx]['latitude'][-1],
        FLT_track[Uidx]['longitude'][-1],
        FLT_track[Uidx]['altitude'][-1]
    )
    
    target_wp = Position(
        GOAL_WPs['latitude'][current_wp_idx],
        GOAL_WPs['longitude'][current_wp_idx],
        current_pos.altitude
    )
    
    # Check if the waypoint is reached
    if _is_waypoint_reached(current_pos, target_wp, optimizer.waypoint_threshold):
        return _advance_to_next_waypoint(FLT_track, FLT_conditions, 
                                       current_wp_idx, len(GOAL_WPs['latitude']))
    
    # Get current flight data
    flight_data = get_current_flight_data(FLT_track, FLT_conditions, nUAVs)
    
    # Generate candidate trajectories
    candidates = _generate_trajectory_candidates(flight_data, Uidx, optimizer)
    
    if not candidates:
        # No valid trajectory found, hold position
        return FLT_track, FLT_conditions, current_wp_idx
    
    # Identify other UAVs (obstacles)
    ObstacleUAVs = np.concatenate([np.arange(Uidx), np.arange(Uidx+1, nUAVs)]).tolist()
    
    # Evaluate safety and select the best trajectory
    safety_flags = _evaluate_trajectory_safety(candidates, flight_data, Uidx, 
                                             ObstacleUAVs, optimizer)

    best_idx = _select_best_trajectory(candidates, safety_flags, target_wp, 
                                     flight_data[Uidx], current_wp_idx, GOAL_WPs)
    
    # Update flight data
    _update_flight_data(FLT_track, FLT_conditions, Uidx, candidates[best_idx])
    
    return FLT_track, FLT_conditions, current_wp_idx

def _is_waypoint_reached(current_pos: Position, target_wp: Position, 
                        threshold: float) -> bool:
    """Checks if the target waypoint is reached"""
    distance = compute_distance(
        {'latitude': current_pos.latitude, 'longitude': current_pos.longitude, 
         'altitude': current_pos.altitude},
        {'latitude': target_wp.latitude, 'longitude': target_wp.longitude, 
         'altitude': target_wp.altitude}
    )[0]
    return distance < threshold

def _advance_to_next_waypoint(FLT_track: Dict, FLT_conditions: Dict, 
                            current_wp_idx: int, total_waypoints: int) -> Tuple[Dict, Dict, int]:
    """Advance to the next waypoint"""
    next_idx = min(current_wp_idx + 1, total_waypoints - 1)
    return FLT_track, FLT_conditions, next_idx

def _generate_trajectory_candidates(flight_data: Dict, uav_idx: int, 
                                  optimizer: TrajectoryOptimizer) -> List[TrajectoryCandidate]:
    """Generates candidate trajectories for the UAV"""
    candidates = []
    uav_data = flight_data[uav_idx]
    
    # Generate possible heading angles
    bearings = _generate_bearing_range(uav_data['bearing'], optimizer.max_turn_rate, 
                                     optimizer.bearing_step)
    
    # Generate possible velocities
    velocities = _generate_velocity_range(uav_data['airspeed'], optimizer.velocity_bounds, 
                                        optimizer.speed_step)
    
    # Generate combinations according to flight mode
    if uav_data['flight_mode'] == 'glide':
        candidates = _generate_glide_trajectories(uav_data, bearings, velocities, optimizer)
    elif uav_data['flight_mode'] == 'engine':
        candidates = _generate_engine_trajectories(uav_data, bearings, velocities, optimizer)
    
    return candidates

def _generate_bearing_range(current_bearing: float, max_turn_rate: float, 
                          step: int) -> List[float]:
    """Generates the range of possible heading angles"""
    left_range = np.linspace(current_bearing - max_turn_rate, current_bearing, step)
    right_range = np.linspace(current_bearing, current_bearing + max_turn_rate, step)
    return np.concatenate([left_range, right_range[1:]]).tolist()

def _generate_velocity_range(current_velocity: float, velocity_bounds: Tuple[float, float], 
                           step: int) -> List[float]:
    """Generates the range of possible velocities"""
    min_vel, max_vel = velocity_bounds
    left_range = np.linspace(min_vel, current_velocity, step)
    right_range = np.linspace(current_velocity, max_vel, step)
    return np.concatenate([left_range, right_range[1:]]).tolist()

def _generate_glide_trajectories(uav_data: Dict, bearings: List[float], 
                               velocities: List[float], 
                               optimizer: TrajectoryOptimizer) -> List[TrajectoryCandidate]:
    """Generates glide mode trajectories"""
    candidates = []
    lb_alt, ub_alt = optimizer.altitude_bounds
    
    for bearing in bearings:
        for velocity in velocities:
            # Compute sink rate
            temp_conditions = {'airspeed': velocity, 'airspeed_dot': 0.0}
            sink_rate = get_sink_rate(optimizer.uav_data, temp_conditions)
            
            # Compute new position
            travel_distance = velocity * optimizer.time_step
            new_lat, new_lon = get_destination_from_range_and_bearing(
                {'latitude': uav_data['latitude'], 'longitude': uav_data['longitude']},
                travel_distance, bearing
            )
            
            altitude_change = -sink_rate * optimizer.time_step
            new_alt = np.clip(uav_data['altitude'] + altitude_change, lb_alt, ub_alt)
            
            candidate = TrajectoryCandidate(
                position=Position(new_lat, new_lon, new_alt),
                bearing=bearing,
                battery_capacity=uav_data['battery_capacity'],  # No consumption in glide
                flight_mode='glide',
                airspeed=velocity,
                flight_path_angle=0.0
            )
            candidates.append(candidate)
    
    return candidates

def _generate_engine_trajectories(uav_data: Dict, bearings: List[float], 
                                velocities: List[float], 
                                optimizer: TrajectoryOptimizer) -> List[TrajectoryCandidate]:
    """Generates engine mode trajectories"""
    candidates = []
    lb_alt, ub_alt = optimizer.altitude_bounds
    
    for bearing in bearings:
        for velocity in velocities:
            # Compute energy consumption
            temp_conditions = {'airspeed': velocity, 'airspeed_dot': 0.0}
            power = get_power_consumption(optimizer.uav_data, temp_conditions)
            power_consumption = power * (optimizer.time_step / 3600)
            
            # Compute new position
            travel_distance = velocity * optimizer.time_step
            new_lat, new_lon = get_destination_from_range_and_bearing(
                {'latitude': uav_data['latitude'], 'longitude': uav_data['longitude']},
                travel_distance, bearing
            )
            
            new_alt = np.clip(uav_data['altitude'] + travel_distance, lb_alt, ub_alt)
            new_battery = uav_data['battery_capacity'] - power_consumption
            
            # Check that battery is not depleted
            if new_battery <= 0:
                continue
            
            candidate = TrajectoryCandidate(
                position=Position(new_lat, new_lon, new_alt),
                bearing=bearing,
                battery_capacity=new_battery,
                flight_mode='engine',
                airspeed=velocity,
                flight_path_angle=0.0
            )
            candidates.append(candidate)
    
    return candidates

def _evaluate_trajectory_safety(candidates: List[TrajectoryCandidate], 
                              flight_data: Dict, uav_idx: int, 
                              obstacle_uavs: List[int], 
                              optimizer: TrajectoryOptimizer) -> List[Tuple[bool, bool]]:
    """Evaluates the safety of each candidate trajectory"""
    safety_flags = []
    
    for candidate in candidates:
        velocity_obstacle_safe = _check_velocity_obstacles(candidate, flight_data, 
                                                         obstacle_uavs, optimizer)
        path_obstacle_safe = _check_path_obstacles(candidate, flight_data, uav_idx, 
                                                 obstacle_uavs, optimizer)
        safety_flags.append((velocity_obstacle_safe, path_obstacle_safe))
    
    return safety_flags

def _check_velocity_obstacles(candidate: TrajectoryCandidate, flight_data: Dict, 
                            obstacle_uavs: List[int], optimizer: TrajectoryOptimizer) -> bool:
    """Checks for velocity obstacles (immediate collision)"""
    for obstacle_idx in obstacle_uavs:
        obstacle_pos = {
            'latitude': flight_data[obstacle_idx]['latitude'],
            'longitude': flight_data[obstacle_idx]['longitude'],
            'altitude': flight_data[obstacle_idx]['altitude']
        }
        
        candidate_pos = {
            'latitude': candidate.position.latitude,
            'longitude': candidate.position.longitude,
            'altitude': candidate.position.altitude
        }
        
        distance = compute_distance(obstacle_pos, candidate_pos)[0]
        
        if distance <= optimizer.horizon_length and distance < optimizer.safe_distance:
            return False
    
    return True

def _check_path_obstacles(candidate: TrajectoryCandidate, flight_data: Dict, 
                        uav_idx: int, obstacle_uavs: List[int], 
                        optimizer: TrajectoryOptimizer) -> bool:
    """Checks for obstacles along the path (future collision)"""
    # Simplified implementation - full logic would require
    # the lineXline function and intersection calculations
    return True  # Placeholder - implement as needed

def _select_best_trajectory(candidates: List[TrajectoryCandidate], 
                          safety_flags: List[Tuple[bool, bool]], 
                          target_wp: Position, uav_data: Dict, 
                          current_wp_idx: int, goal_wps: Dict) -> int:
    """Selects the best trajectory based on multiple criteria"""
    
    if not candidates:
        return 0
    
    # Compute costs for each criterion
    safety_costs = []
    distance_costs = []
    energy_costs = []
    altitude_costs = []
    
    for i, (candidate, (vo_safe, po_safe)) in enumerate(zip(candidates, safety_flags)):
        # Safety cost (negative if safe)
        safety_costs.append(-float(vo_safe and po_safe))
        
        # Distance cost to waypoint
        target_dict = {
            'latitude': target_wp.latitude,
            'longitude': target_wp.longitude,
            'altitude': target_wp.altitude
        }
        candidate_dict = {
            'latitude': candidate.position.latitude,
            'longitude': candidate.position.longitude,
            'altitude': candidate.position.altitude
        }
        distance = compute_distance(candidate_dict, target_dict)[0]
        distance_costs.append(distance)
        
        # Energy cost
        energy_ratio = (uav_data['battery_capacity'] - candidate.battery_capacity) / uav_data['battery_capacity']
        energy_costs.append(energy_ratio)
        
        # Altitude cost (prefer to maintain altitude)
        altitude_ratio = abs(uav_data['altitude'] - candidate.position.altitude) / max(uav_data['altitude'], 1)
        altitude_costs.append(altitude_ratio)
    
    # Normalize costs
    def normalize_costs(costs):
        costs_array = np.array(costs)
        norm = np.linalg.norm(costs_array)
        if norm == 0:
            return np.zeros_like(costs_array)
        normalized = costs_array / norm
        normalized[np.isnan(normalized)] = 0
        return normalized
    
    norm_safety = normalize_costs(safety_costs)
    norm_distance = normalize_costs(distance_costs)
    norm_energy = normalize_costs(energy_costs)
    norm_altitude = normalize_costs(altitude_costs)
    
    # Compute total cost (weighted sum)
    total_costs = norm_safety + norm_distance + norm_energy + norm_altitude
    
    # Return the index of the minimal cost
    return int(np.argmin(total_costs))

def _update_flight_data(FLT_track: Dict, FLT_conditions: Dict, 
                       uav_idx: int, best_candidate: TrajectoryCandidate):
    """Updates flight data with the best trajectory"""
    
    # Update flight history
    FLT_track[uav_idx]['latitude'].append(best_candidate.position.latitude)
    FLT_track[uav_idx]['longitude'].append(best_candidate.position.longitude)
    FLT_track[uav_idx]['altitude'].append(best_candidate.position.altitude)
    FLT_track[uav_idx]['bearing'].append(best_candidate.bearing)
    FLT_track[uav_idx]['flight_mode'].append(best_candidate.flight_mode)
    FLT_track[uav_idx]['battery_capacity'].append(best_candidate.battery_capacity)
    
    # Update flight conditions
    FLT_conditions[uav_idx]['airspeed'] = best_candidate.airspeed
    FLT_conditions[uav_idx]['flight_path_angle'] = best_candidate.flight_path_angle
