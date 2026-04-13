import numpy as np
import threading
from typing import Dict, Optional

from compute import point_in_polygon

class Thermal:
    """Classe représentant un thermique"""
    
    def __init__(self, x: float, y: float, radius: float, strength: float, duration: float, start_time: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.strength = strength  # Vitesse de montée en m/s
        self.duration = duration  # Durée en secondes
        self.start_time = start_time
        self.active = True
        
    def is_active(self, current_time: float) -> bool:
        """Vérifie si le thermique est encore actif"""
        return self.active and (current_time - self.start_time) < self.duration
    
    def get_lift_rate(self, distance_from_center: float) -> float:
        """Calcule la vitesse de montée à une distance donnée du centre"""
        
        if distance_from_center > self.radius:
            return 0.0
        
        # Profil de vitesse gaussien avec maximum au centre
        normalized_distance = distance_from_center / self.radius
        lift_factor = np.exp(-2 * normalized_distance**2)
        
        return self.strength * lift_factor

class ThermalMap:
    """
    Carte partagée et **thread-safe** des thermiques.

    Source de vérité unique :
      – le thread ROS2 écrit via ``update_from_snapshot`` / ``remove_thermal``
      – le thread asyncio principal lit via ``get_active_thermals`` / ``get_info``
        et écrit l'évaluation via ``mark_evaluated``.

    Structure interne (_entries) :
        {thermal_id: {'thermal': Thermal,
                      'detection_time': float,
                      'detected':       bool,   # un drone l'a physiquement traversée
                      'evaluated':      bool,
                      'alt_gain':       float}}
    """

    def __init__(self):
        self._entries: Dict[int, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Bridge API (appelé depuis le thread ROS2)
    # ------------------------------------------------------------------
    def update_from_snapshot(self, new_thermals: Dict[int, 'Thermal'], now: float):
        """
        Mise à jour atomique depuis un snapshot ROS.
        - Ajoute les nouvelles thermiques.
        - Met à jour l'objet Thermal des thermiques existantes (position,
          force, rayon…) tout en **conservant** le statut d'évaluation.
        - Supprime les thermiques qui ne sont plus dans le snapshot.
        """
        with self._lock:
            for tid, th in new_thermals.items():
                if tid in self._entries:
                    # Mettre à jour l'objet Thermal (position peut bouger)
                    self._entries[tid]['thermal'] = th
                else:
                    self._entries[tid] = {
                        'thermal': th,
                        'detection_time': now,
                        'detected': False,
                        'evaluated': False,
                        'alt_gain': 0.0,
                    }
            stale = [k for k in self._entries if k not in new_thermals]
            for k in stale:
                del self._entries[k]

    def remove_thermal(self, thermal_id: int):
        """Retirer explicitement un thermique (callback /thermal_removed)."""
        with self._lock:
            self._entries.pop(thermal_id, None)

    # ------------------------------------------------------------------
    # Main-loop API (appelé depuis le thread asyncio)
    # ------------------------------------------------------------------
    def get_active_thermals(self) -> Dict[int, 'Thermal']:
        """Retourne un snapshot thread-safe ``{tid: Thermal}`` (toutes)."""
        with self._lock:
            return {tid: e['thermal'] for tid, e in self._entries.items()}

    def get_detected_thermals(self) -> Dict[int, 'Thermal']:
        """Retourne uniquement les thermiques qu'un drone a physiquement
        traversées (``detected=True``)."""
        with self._lock:
            return {tid: e['thermal'] for tid, e in self._entries.items()
                    if e['detected']}

    def get_info(self, thermal_id: int) -> Optional[dict]:
        """
        Retourne les métadonnées d'évaluation, ou None si la thermique est
        absente.  Le dict retourné est une **copie** (pas de mutation
        concurrente possible).
        """
        with self._lock:
            entry = self._entries.get(thermal_id)
            if entry is None:
                return None
            return {
                'detected': entry['detected'],
                'evaluated': entry['evaluated'],
                'alt_gain': entry['alt_gain'],
                'detection_time': entry['detection_time'],
            }

    def get_thermal_obj(self, thermal_id: int) -> Optional['Thermal']:
        """Retourne l'objet Thermal ou None."""
        with self._lock:
            entry = self._entries.get(thermal_id)
            return entry['thermal'] if entry else None

    def mark_detected(self, thermal_id: int):
        """Marquer une thermique comme physiquement détectée par un drone."""
        with self._lock:
            if thermal_id in self._entries:
                self._entries[thermal_id]['detected'] = True

    def mark_evaluated(self, thermal_id: int, evaluated: bool = True,
                       alt_gain: float = 0.0):
        """Mettre à jour le statut d'évaluation d'un thermique."""
        with self._lock:
            if thermal_id in self._entries:
                self._entries[thermal_id]['evaluated'] = evaluated
                self._entries[thermal_id]['alt_gain'] = alt_gain

    def ensure_exists(self, thermal_id: int, thermal: 'Thermal', now: float):
        """
        S'assurer qu'un thermique est dans la carte (ajout si absent).
        Utile en fallback si un thermique est détecté avant l'arrivée du
        prochain snapshot ROS.
        """
        with self._lock:
            if thermal_id not in self._entries:
                self._entries[thermal_id] = {
                    'thermal': thermal,
                    'detection_time': now,
                    'detected': True,
                    'evaluated': False,
                    'alt_gain': 0.0,
                }
            else:
                # Si déjà présent (via snapshot ROS), marquer comme détecté
                self._entries[thermal_id]['detected'] = True

    def __len__(self):
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Rétro-compatibilité  (préférer get_info / get_active_thermals)
    # ------------------------------------------------------------------
    def change_thermal_status(self, thermal_id: int, evaluated: bool = False,
                              alt_gain = 0.0):
        """Ancienne API — redirige vers mark_evaluated."""
        self.mark_evaluated(thermal_id, evaluated, alt_gain)


class ThermalGenerator:
    """Générateur de thermiques avec paramètres variables"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.thermals = dict()
        
    def generate_random_thermals(self, num_thermals: int, obstacles: Dict, current_time: float, start_id: int = 0) -> Dict:
        """Génère des thermiques aléatoires sur la carte"""
        
        for i in range(num_thermals):
            # Position aléatoire dans les limites de la carte
            x = np.random.uniform(self.params['X_lower_bound'], self.params['X_upper_bound'])
            y = np.random.uniform(self.params['Y_lower_bound'], self.params['Y_upper_bound'])
            
            # Paramètres variables
            radius = np.random.uniform(150, 300)  # Rayon entre 150 et 300m
            strength = np.random.uniform(2.0, 5.0)  # Vitesse de montée entre 2.0 et 5.0 m/s
            duration = np.random.uniform(360, 600)  # Durée entre 6 et 10 minutes
            
            # vérifier que le thermique n'est pas généré à l'intérieur d'un obstacle
            inside_obstacle = False
            for obstacle in obstacles:
                if point_in_polygon((x, y), obstacle['vertices']):
                    inside_obstacle = True
                    break
            
            if inside_obstacle:
                continue  # Regénérer ce thermique
            
            thermal_id = i + start_id
            i += 1
            
            thermal = Thermal(x, y, radius, strength, duration, current_time)
            self.thermals[thermal_id] = thermal
            
        return self.thermals

def detect_thermal_at_position(position: Dict, thermals: Dict, current_time: float) -> int:
    """
    Détecte les thermiques seulement quand le UAV est dans la zone du thermique (variation de vent)
    
    Args:
        position (Dict): Position actuelle du UAV {'X': float, 'Y': float, 'Z': float}
        thermals (Dict): Dictionnaire des thermiques actifs
        current_time (float): Temps de simulation actuel
        
    Returns:
        int: ID du thermique détecté, ou None si aucun thermique n'est détecté
    """
    for thermal_id, thermal in thermals.items():
        if not thermal.is_active(current_time):
            continue
            
        # Calculer la distance horizontale du UAV au centre du thermique
        distance = np.sqrt((position['X'] - thermal.x)**2 + (position['Y'] - thermal.y)**2)
        
        # Détection seulement si le UAV est DANS la zone du thermique
        if distance <= thermal.radius:
            # Vérifier aussi que le UAV peut sentir la variation de vent (lift_rate > 0)
            lift_rate = thermal.get_lift_rate(distance)
            if lift_rate > 0.1:  # Seuil minimum pour détecter la variation de vent
                return thermal_id

    return None


class ThermalEvaluator:
    """Évaluateur de thermiques pour les UAVs"""
    
    def __init__(self, params: Dict, UAV_data: Dict):
        self.params = params
        self.UAV_data = UAV_data
        self.evaluation_radius = 50  # Rayon d'évaluation réduit (en mètres)
        self.min_evaluation_time = 30  # Temps minimum d'évaluation en secondes
        self.max_soaring_time = 600  # Temps maximum d'exploitation en secondes (10 minutes)
        self.min_separation_distance = 50  # Distance minimale entre UAVs dans un thermique (en mètres)
        self.min_climb_rate = 0.3  # Taux de montée minimum en m/s pour considérer un thermique profitable

