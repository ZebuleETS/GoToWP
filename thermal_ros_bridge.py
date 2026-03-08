"""
ROS2 ↔ Algorithm Bridge for Gazebo Thermals
============================================
Subscribes to the thermal_generator_node ROS2 topics and converts
Gazebo thermals into thermal.py Thermal objects usable by dronePx4.py.

Usage in dronePx4.py:
    from thermal_ros_bridge import ThermalROSBridge

    bridge = ThermalROSBridge(origin_lat, origin_lon)
    bridge.start()          # launches ROS2 spin in a background thread

    # In the main loop:
    active = bridge.get_active_thermals()       # {thermal_id: Thermal}
    tmap   = bridge.get_thermal_map()           # shared ThermalMap
    bridge.stop()
"""

import math
import threading
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from thermal import Thermal, ThermalMap

# Must match thermal_generator_node.py
FIELDS_PER_THERMAL = 9

# Earth radius for coordinate conversions
EARTH_RADIUS = 6371000.0


def _gps_to_enu(lat, lon, origin_lat, origin_lon):
    """Convert GPS (lat, lon) to ENU (x_east, y_north) relative to an origin."""
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    lat_avg = math.radians((lat + origin_lat) / 2.0)
    x = EARTH_RADIUS * dlon * math.cos(lat_avg)
    y = EARTH_RADIUS * dlat
    return x, y


class _ThermalBridgeNode(Node):
    """Internal ROS2 node that subscribes to thermal topics."""

    def __init__(self, bridge: 'ThermalROSBridge'):
        super().__init__('thermal_bridge_node')
        self._bridge = bridge

        # Subscribe to the full snapshot (all active thermals each cycle)
        self.create_subscription(
            Float32MultiArray, '/thermal_snapshot',
            self._on_snapshot, 10)

        # Subscribe to removed IDs for immediate cleanup
        self.create_subscription(
            Float32MultiArray, '/thermal_removed',
            self._on_removed, 10)

        self.get_logger().info("ThermalBridgeNode started — listening on /thermal_snapshot & /thermal_removed")

    # ------------------------------------------------------------------ #
    def _on_snapshot(self, msg: Float32MultiArray):
        """Receive the full set of active thermals from the generator."""
        data = msg.data
        if len(data) % FIELDS_PER_THERMAL != 0:
            self.get_logger().warn(
                f"Snapshot length {len(data)} not a multiple of {FIELDS_PER_THERMAL}")
            return

        new_thermals: Dict[int, Thermal] = {}
        for i in range(0, len(data), FIELDS_PER_THERMAL):
            tid      = int(data[i])
            lon      = data[i + 1]
            lat      = data[i + 2]
            x_enu    = data[i + 3]
            y_enu    = data[i + 4]
            radius   = data[i + 5]
            strength = data[i + 6]
            lifetime = data[i + 7]
            birth    = data[i + 8]

            # If the bridge has its own origin, re-compute ENU from GPS
            if self._bridge.origin_lat is not None:
                x_enu, y_enu = _gps_to_enu(
                    lat, lon,
                    self._bridge.origin_lat, self._bridge.origin_lon)

            # Use start_time=0 and duration=inf so that is_active()
            # always returns True for bridge-managed thermals.
            # The bridge removes expired thermals when they disappear
            # from the /thermal_snapshot topic, so is_active() should
            # never gate detection for ROS-sourced thermals.
            thermal = Thermal(
                x=x_enu,
                y=y_enu,
                radius=radius,
                strength=strength,
                duration=float('inf'),
                start_time=0.0,
            )
            # Attach GPS coords and original timing as extra attributes
            thermal.lat = lat
            thermal.lon = lon
            thermal.gz_birth_time = birth
            thermal.gz_lifetime = lifetime
            thermal.ros_managed = True

            new_thermals[tid] = thermal

        # Atomic swap of the thermal dict
        now = time.time()
        with self._bridge._lock:
            self._bridge._last_update = now

        # Update the single source of truth (ThermalMap)
        # update_from_snapshot is internally lock-protected
        self._bridge._thermal_map.update_from_snapshot(new_thermals, now)

    # ------------------------------------------------------------------ #
    def _on_removed(self, msg: Float32MultiArray):
        """Explicitly remove expired thermals."""
        with self._bridge._lock:
            for fid in msg.data:
                tid = int(fid)
                self._bridge._thermal_map.remove_thermal(tid)


class ThermalROSBridge:
    """
    High-level bridge between ROS2 thermal generator and the trajectory
    planning algorithm in dronePx4.py.

    Thread-safe: the ROS2 subscriber runs in a background thread.
    """

    def __init__(self, origin_lat: Optional[float] = None,
                 origin_lon: Optional[float] = None):
        """
        Args:
            origin_lat/lon: ENU origin used by dronePx4.py.  If provided
                the bridge re-computes ENU positions from GPS (accounting
                for multi-drone shared origin).  If None, uses the ENU
                coordinates from the generator directly.
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon

        self._thermal_map = ThermalMap()
        self._lock = threading.Lock()       # protège uniquement _last_update
        self._last_update: float = 0.0

        self._node: Optional[_ThermalBridgeNode] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        """Initialise ROS2 (if needed) and start background spin thread."""
        if self._running:
            return
        if not rclpy.ok():
            rclpy.init()
        self._node = _ThermalBridgeNode(self)
        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True, name="ros2_thermal_bridge")
        self._spin_thread.start()
        print("[ThermalROSBridge] Started — waiting for thermals from /thermal_snapshot")

    def stop(self):
        """Shutdown the bridge cleanly."""
        self._running = False
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        # Don't call rclpy.shutdown() here — other nodes may still be running

    def _spin_loop(self):
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    # ------------------------------------------------------------------ #
    # Public API for dronePx4.py
    # ------------------------------------------------------------------ #
    def get_active_thermals(self) -> Dict[int, 'Thermal']:
        """
        Return currently active thermals as a dict {thermal_id: Thermal}.
        Thread-safe snapshot from the ThermalMap (single source of truth).
        """
        return self._thermal_map.get_active_thermals()

    def get_thermal_map(self) -> ThermalMap:
        """Return the shared ThermalMap (same instance, updated live)."""
        return self._thermal_map

    def has_thermals(self) -> bool:
        return len(self._thermal_map) > 0

    def seconds_since_last_update(self) -> float:
        if self._last_update == 0.0:
            return float('inf')
        return time.time() - self._last_update

    def wait_for_thermals(self, timeout: float = 30.0) -> bool:
        """Block until at least one thermal is received or timeout expires."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.has_thermals():
                return True
            time.sleep(0.5)
        return False

    def set_origin(self, lat: float, lon: float):
        """Update the ENU origin (call after getting shared drone home)."""
        self.origin_lat = lat
        self.origin_lon = lon
