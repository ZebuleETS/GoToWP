"""
Scénarios de test pour évaluation multi-UAV avec PX4 SITL
Tests d'optimalité, d'endurance et de couverture
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json
import csv
import os
from collections import Counter

from trajectory import generate_random_obstacles


class TestScenario(Enum):
    """Types de scénarios de test"""
    PRELIMINARY_COLLISION = "preliminary_collision"
    TRAJECTORY_OPTIMAL_POWERED = "trajectory_optimal_powered"
    TRAJECTORY_OPTIMAL_GLIDE = "trajectory_optimal_glide"
    ENDURANCE = "endurance"
    COVERAGE = "coverage"


@dataclass
class PerformanceMetrics:
    """Métriques de performance pour analyse"""
    scenario_name: str
    num_uavs: int
    
    # Métriques de trajectoire
    total_distance: Dict[int, float] = field(default_factory=dict)
    path_length: Dict[int, float] = field(default_factory=dict)
    collisions_avoided: int = 0
    min_separation_distance: float = float('inf')
    
    # Métriques d'énergie
    battery_consumed: Dict[int, float] = field(default_factory=dict)
    battery_remaining: Dict[int, float] = field(default_factory=dict)
    
    # Métriques de temps
    total_flight_time: Dict[int, float] = field(default_factory=dict)
    glide_time: Dict[int, float] = field(default_factory=dict)
    soar_time: Dict[int, float] = field(default_factory=dict)
    engine_time: Dict[int, float] = field(default_factory=dict)
    
    # Métriques thermiques (endurance)
    thermals_generated: int = 0
    thermals_detected: int = 0
    thermals_exploited: int = 0
    thermals_rejected: int = 0
    thermals_per_uav: Dict[int, int] = field(default_factory=dict)
    
    # Métriques d'endurance (patrouille)
    patrol_loops: Dict[int, int] = field(default_factory=dict)
    soaring_ratio: Dict[int, float] = field(default_factory=dict)  # soar_time / total_time

    # Métriques complémentaires (M1..M7)
    motor_off_ratio: Dict[int, float] = field(default_factory=dict)  # (glide+soar)/total en %
    battery_consumption_rate_ah_per_h: Dict[int, float] = field(default_factory=dict)
    potential_energy_j: Dict[int, Dict[str, float]] = field(default_factory=dict)
    thermals_detected_per_uav: Dict[int, int] = field(default_factory=dict)
    exploited_detected_ratio: Dict[int, float] = field(default_factory=dict)
    thermal_exploitation_frequency: Dict[int, int] = field(default_factory=dict)
    thermal_exploitation_duration_s: Dict[int, float] = field(default_factory=dict)
    thermal_exploitation_duration_per_uav_s: Dict[int, Dict[int, float]] = field(default_factory=dict)
    thermal_exploited_unique: int = 0
    thermal_exploitation_global_ratio: float = 0.0  # eta_exp,global = N_exp / N_total en %

    # Métriques sécurité/performance calcul (M8..M11)
    algorithm_time_avg_ms: float = 0.0
    algorithm_time_max_ms: float = 0.0
    
    # Métriques de couverture
    objects_total: int = 0
    objects_detected: int = 0
    objects_per_uav: Dict[int, int] = field(default_factory=dict)
    detection_rate: float = 0.0


@dataclass
class SurveillanceObject:
    """Objet de surveillance à détecter"""
    id: int
    x: float
    y: float
    z: float
    spawn_time: float
    despawn_time: float
    detected: bool = False
    detected_by: List[int] = field(default_factory=list)
    
    def is_active(self, current_time: float) -> bool:
        """Vérifie si l'objet est actif"""
        return self.spawn_time <= current_time <= self.despawn_time
    
    def is_in_fov(self, uav_x: float, uav_y: float, fov_radius: float) -> bool:
        """Vérifie si l'objet est dans le champ de vision"""
        distance = np.sqrt((self.x - uav_x)**2 + (self.y - uav_y)**2)
        return distance <= fov_radius


class ScenarioGenerator:
    """Générateur de scénarios de test"""
    
    @staticmethod
    def generate_preliminary_collision_scenario(nUAVs: int, params: dict, home_positions: dict) -> Tuple[dict, dict, list]:
        """
        Scénario préliminaire : collision garantie à 100%
        Tous les drones convergent vers le même point avec obstacles au milieu
        """
        print("\n" + "="*70)
        print("GÉNÉRATION SCÉNARIO: Collision Préliminaire (100% risque)")
        print("="*70)
        
        # Destination commune éloignée du home (0,0)
        # Le home est au centre de la zone → décaler la cible vers le quadrant positif
        target_x = params['X_upper_bound'] * 0.65
        target_y = params['Y_upper_bound'] * 0.65
        target_z = 400.0
        
        num_vertices = np.random.randint(4, 6)  # Nombre de sommets pour le polygone
        radius = np.random.uniform(100, 500)
        vertices = []
        for _ in range(num_vertices):
            angle = np.random.uniform(0, 2 * np.pi)
            x = target_x + radius * np.cos(angle)
            y = target_y + radius * np.sin(angle)
            vertices.append((x, y))
        center_obstacle = {
            'vertices': vertices,
        }
        print(f"✓ Obstacle central généré avec {num_vertices} sommets et rayon ~{radius:.0f}m")
        
        obstacles = [center_obstacle]
        
        start_positions = {u: home_positions[u] for u in range(nUAVs)}
        
        end_position = {
            'X': target_x,
            'Y': target_y,
            'Z': target_z
        }
        
        print(f"\n✓ Scénario collision: convergence vers ({target_x:.0f}, {target_y:.0f}, {target_z:.0f})")
        
        return start_positions, end_position, obstacles
    
    @staticmethod
    def generate_trajectory_optimal_scenario(nUAVs: int, params: dict, home_positions: dict, allow_glide: bool) -> Tuple[dict, dict, dict]:
        """
        Scénarios 1 & 2 : Trajectoires optimales
        allow_glide=False : moteur allumé (scénario 1)
        allow_glide=True : planeur autorisé (scénario 2)
        """
        scenario_name = "Trajectoire Optimale (Planeur)" if allow_glide else "Trajectoire Optimale (Propulsé)"
        print("\n" + "="*70)
        print(f"GÉNÉRATION SCÉNARIO: {scenario_name}")
        print("="*70)
        
        # Zone de test délimitée
        test_zone = {
            'x_min': params['X_lower_bound'] + 500,
            'x_max': params['X_upper_bound'] - 500,
            'y_min': params['Y_lower_bound'] + 500,
            'y_max': params['Y_upper_bound'] - 500
        }
        
        # Destination commune éloignée du home (0,0)
        target_x = params['X_upper_bound'] * 0.65
        target_y = params['Y_upper_bound'] * 0.65
        target_z = 400.0
        
        # Générer obstacles variables (plus pour scénario 2)
        num_obstacles = 8 if allow_glide else 5
        obstacles = generate_random_obstacles(num_obstacles, params)
        
        # Positions de départ (identiques ou non)
        start_positions = {u: home_positions[u] for u in range(nUAVs)}
        
        end_position = {
            'X': target_x,
            'Y': target_y,
            'Z': target_z
        }
        
        print(f"✓ Mode: {'Planeur autorisé' if allow_glide else 'Propulsion continue'}")
        print(f"✓ Destination commune: ({target_x:.0f}, {target_y:.0f})")
        print(f"✓ {num_obstacles} obstacles ")
        
        return start_positions, end_position, obstacles
    
    @staticmethod
    def generate_endurance_scenario(nUAVs: int, params: dict, thermal_generator, obstacles) -> Tuple[dict, dict]:
        """
        Scénario d'endurance : évaluation exploitation thermiques
        """
        print("\n" + "="*70)
        print("GÉNÉRATION SCÉNARIO: Test d'Endurance")
        print("="*70)
        
        # Générer thermiques variées
        num_thermals = np.random.randint(8, 15)
        
        # Créer thermiques avec paramètres variables
        thermals = thermal_generator.generate_random_thermals(num_thermals, obstacles, params['current_simulation_time'])
        
        # Statistiques thermiques
        thermal_stats = {
            'total': len(thermals),
            'weak': sum(1 for t in thermals.values() if t.get_strength() < 2.5),
            'medium': sum(1 for t in thermals.values() if 2.5 <= t.get_strength() < 3.5),
            'strong': sum(1 for t in thermals.values() if t.get_strength() >= 3.5)
        }
        
        print(f"✓ {num_thermals} thermiques générées")
        print(f"  - Faibles (<2.5 m/s): {thermal_stats['weak']}")
        print(f"  - Moyennes (2.5-3.5 m/s): {thermal_stats['medium']}")
        print(f"  - Fortes (>3.5 m/s): {thermal_stats['strong']}")
        print(f"✓ Durée de vie: 5-15 min")
        print(f"✓ Rayon: 80-200m")
        
        return thermals, thermal_stats
    
    @staticmethod
    def generate_coverage_scenario(nUAVs: int, params: dict, mission_duration: float) -> List[SurveillanceObject]:
        """
        Scénario de couverture : surveillance d'objets/événements
        """
        print("\n" + "="*70)
        print("GÉNÉRATION SCÉNARIO: Test de Couverture")
        print("="*70)
        
        # Générer davantage d'objets avec fort chevauchement temporel
        # pour augmenter le nombre d'objets actifs en simultané.
        num_min = int(params.get('coverage_num_objects_min', 35))
        num_max = int(params.get('coverage_num_objects_max', 55))
        if num_max <= num_min:
            num_max = num_min + 1
        num_objects = np.random.randint(num_min, num_max)

        # Les apparitions sont concentrées dans une fenêtre plus courte.
        spawn_window_ratio = float(params.get('coverage_spawn_window_ratio', 0.45))
        spawn_window_ratio = max(0.05, min(1.0, spawn_window_ratio))
        spawn_window = mission_duration * spawn_window_ratio

        # Une partie des objets apparaît tôt et reste active longtemps.
        hot_start_ratio = float(params.get('coverage_hot_start_ratio', 0.30))
        hot_start_ratio = max(0.0, min(1.0, hot_start_ratio))
        hot_start_count = int(num_objects * hot_start_ratio)
        hot_start_window = mission_duration * 0.08

        lifetime_min = float(params.get('coverage_lifetime_min_s', 700.0))
        lifetime_max = float(params.get('coverage_lifetime_max_s', 1400.0))
        if lifetime_max < lifetime_min:
            lifetime_max = lifetime_min

        objects = []
        events = []
        
        for i in range(num_objects):
            if i < hot_start_count:
                spawn_time = np.random.uniform(0, hot_start_window)
                lifetime = np.random.uniform(lifetime_min * 1.2, lifetime_max * 1.5)
            else:
                spawn_time = np.random.uniform(0, spawn_window)
                lifetime = np.random.uniform(lifetime_min, lifetime_max)

            despawn_time = spawn_time + lifetime
            
            obj = SurveillanceObject(
                id=i,
                x=np.random.uniform(-1500, 1500),
                y=np.random.uniform(-1500, 1500),
                z=0,
                spawn_time=spawn_time,
                despawn_time=despawn_time
            )
            objects.append(obj)

            events.append((spawn_time, 1))
            events.append((despawn_time, -1))

        # Estimation simple du pic d'objets actifs simultanément.
        active_count = 0
        peak_active = 0
        for _, delta in sorted(events, key=lambda e: (e[0], -e[1])):
            active_count += delta
            peak_active = max(peak_active, active_count)
        
        print(f"✓ {num_objects} objets de surveillance générés")
        print(f"✓ Fenêtre d'apparition: 0-{spawn_window:.0f}s ({spawn_window_ratio*100:.0f}% mission)")
        print(f"✓ Durée de vie: {lifetime_min/60:.1f}-{lifetime_max/60:.1f} min")
        print(f"✓ Objets early persistants: {hot_start_count}")
        print(f"✓ Pic estimé objets actifs simultanés: {peak_active}")
        print(f"✓ Distribution spatiale: aléatoire")
        print(f"✓ Champ de vision UAV: 150m")
        
        return objects


class PerformanceAnalyzer:
    """Analyseur de performances"""
    
    @staticmethod
    def calculate_flight_phase_times(FLT_track: dict, nUAVs: int) -> Dict[int, Dict[str, float]]:
        """Calculer les temps de chaque phase de vol"""
        phase_times = {}
        
        for u in range(nUAVs):
            glide_time = 0
            soar_time = 0
            engine_time = 0
            
            if len(FLT_track[u]['flight_mode']) > 1:
                for i in range(1, len(FLT_track[u]['flight_mode'])):
                    time_step = FLT_track[u]['flight_time'][i] - FLT_track[u]['flight_time'][i-1]
                    mode = FLT_track[u]['flight_mode'][i]
                    
                    if mode == 'glide':
                        glide_time += time_step
                    elif mode == 'soaring':
                        soar_time += time_step
                    elif mode == 'engine':
                        engine_time += time_step
            
            phase_times[u] = {
                'glide': glide_time,
                'soar': soar_time,
                'engine': engine_time,
                'total': FLT_track[u]['flight_time'][-1] if len(FLT_track[u]['flight_time']) > 0 else 0
            }
        
        return phase_times
    
    @staticmethod
    def calculate_path_metrics(FLT_track: dict, nUAVs: int) -> Dict[int, Dict[str, float]]:
        """Calculer les métriques de trajectoire"""
        path_metrics = {}
        
        for u in range(nUAVs):
            total_distance = 0
            if len(FLT_track[u]['X']) > 1:
                for i in range(1, len(FLT_track[u]['X'])):
                    dx = FLT_track[u]['X'][i] - FLT_track[u]['X'][i-1]
                    dy = FLT_track[u]['Y'][i] - FLT_track[u]['Y'][i-1]
                    dz = FLT_track[u]['Z'][i] - FLT_track[u]['Z'][i-1]
                    total_distance += np.sqrt(dx**2 + dy**2 + dz**2)
            
            path_metrics[u] = {
                'distance': total_distance,
                'waypoints': len(FLT_track[u]['X'])
            }
        
        return path_metrics
    
    @staticmethod
    def check_min_separation(FLT_track: dict, nUAVs: int) -> float:
        """Calculer la distance minimale de séparation entre UAVs"""
        min_sep = float('inf')
        
        # Trouver la longueur minimale
        min_len = min(len(FLT_track[u]['X']) for u in range(nUAVs))
        
        for i in range(min_len):
            for u1 in range(nUAVs):
                for u2 in range(u1 + 1, nUAVs):
                    dx = FLT_track[u1]['X'][i] - FLT_track[u2]['X'][i]
                    dy = FLT_track[u1]['Y'][i] - FLT_track[u2]['Y'][i]
                    dz = FLT_track[u1]['Z'][i] - FLT_track[u2]['Z'][i]
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    min_sep = min(min_sep, distance)
        
        return min_sep
    
    @staticmethod
    def analyze_thermal_exploitation(thermals: list, FLT_track: dict, nUAVs: int) -> dict:
        """Analyser l'exploitation des thermiques"""
        detected = 0
        rejected = 0
        for t in thermals:
            if isinstance(t, dict):
                if t.get('detected', False):
                    detected += 1
                if t.get('rejected', False):
                    rejected += 1
            else:
                # Objet Thermal : compter comme détecté (actif dans la simulation)
                detected += 1
        
        # Compter par UAV
        thermals_per_uav = {}
        for u in range(nUAVs):
            count = 0
            for t in thermals:
                if isinstance(t, dict):
                    if u in t.get('exploited_by', []):
                        count += 1
                else:
                    # Vérifier si cet UAV a visité cette thermique via FLT_track
                    tid = getattr(t, 'id', None)
                    if tid is not None and tid in FLT_track.get(u, {}).get('visited_thermals', set()):
                        count += 1
            thermals_per_uav[u] = count

        exploited = sum(thermals_per_uav.values())
        
        return {
            'detected': detected,
            'exploited': exploited,
            'rejected': rejected,
            'per_uav': thermals_per_uav
        }

    @staticmethod
    def calculate_potential_energy_variation(FLT_track: dict, nUAVs: int,
                                            uav_mass: float = 1.6,
                                            g: float = 9.81) -> Dict[int, Dict[str, float]]:
        """Retourne les métriques d'énergie potentielle (Joules) par UAV.

        - thermal_gain_J: somme des gains d'altitude en vol non-moteur
        - engine_gain_J: somme des gains d'altitude en mode moteur
        - gain_J: somme des gains d'altitude sur tout le vol (dz > 0)
        - loss_J: somme des pertes d'altitude sur tout le vol (|dz| pour dz < 0)
        - variation_J: variation nette entre début et fin du vol
        """
        energy = {}
        for u in range(nUAVs):
            z = FLT_track[u].get('Z', [])
            modes = FLT_track[u].get('flight_mode', [])
            if len(z) < 2:
                energy[u] = {
                    'variation_J': 0.0,
                    'thermal_gain_J': 0.0,
                    'engine_gain_J': 0.0,
                    'gain_J': 0.0,
                    'loss_J': 0.0,
                }
                continue

            gain_alt = 0.0
            loss_alt = 0.0
            thermal_gain_alt = 0.0
            engine_gain_alt = 0.0
            for i in range(1, len(z)):
                dz = z[i] - z[i - 1]
                if dz > 0.0:
                    gain_alt += dz
                    # Attribution par source:
                    # - engine -> gain moteur
                    # - tout autre mode (soaring/glide) -> gain thermique/non-moteur
                    if i < len(modes) and modes[i] == 'engine':
                        engine_gain_alt += dz
                    else:
                        thermal_gain_alt += dz
                elif dz < 0.0:
                    loss_alt += -dz

            variation_alt = z[-1] - z[0]
            thermal_gain_j = uav_mass * g * thermal_gain_alt
            engine_gain_j = uav_mass * g * engine_gain_alt
            gain_j = uav_mass * g * gain_alt
            loss_j = uav_mass * g * loss_alt
            variation_j = uav_mass * g * variation_alt

            energy[u] = {
                'variation_J': variation_j,
                'thermal_gain_J': thermal_gain_j,
                'engine_gain_J': engine_gain_j,
                'gain_J': gain_j,
                'loss_J': loss_j,
            }
        return energy

    @staticmethod
    def calculate_thermal_exploitation_log_metrics(FLT_track: dict, nUAVs: int) -> dict:
        """Calcule fréquence et durées d'exploitation par thermique depuis les logs UAV."""
        frequency = Counter()
        duration_per_thermal = Counter()
        duration_per_uav = {u: {} for u in range(nUAVs)}
        total_exploitations = 0

        for u in range(nUAVs):
            entries = FLT_track[u].get('thermal_exploitation_log', [])
            for entry in entries:
                tid = entry.get('thermal_id')
                if tid is None:
                    continue
                frequency[tid] += 1
                total_exploitations += 1
                start_t = entry.get('entry_time')
                exit_t = entry.get('exit_time')
                if start_t is None or exit_t is None:
                    continue
                duration = max(0.0, exit_t - start_t)
                duration_per_thermal[tid] += duration
                duration_per_uav[u][tid] = duration_per_uav[u].get(tid, 0.0) + duration

        return {
            'frequency': dict(frequency),
            'duration_per_thermal_s': dict(duration_per_thermal),
            'duration_per_uav_s': duration_per_uav,
            'total_exploitations': total_exploitations,
        }
    
    @staticmethod
    def analyze_coverage(objects: List[SurveillanceObject], FLT_track: dict, 
                        nUAVs: int, fov_radius: float) -> dict:
        """Analyser la couverture des objets"""
        
        # Vérifier détection pour chaque objet
        for obj in objects:
            for u in range(nUAVs):
                if len(FLT_track[u]['X']) == 0:
                    continue
                
                # Parcourir la trajectoire
                for i in range(len(FLT_track[u]['X'])):
                    current_time = FLT_track[u]['flight_time'][i]
                    
                    # Vérifier si objet actif
                    if not obj.is_active(current_time):
                        continue
                    
                    # Vérifier si dans le champ de vision
                    if obj.is_in_fov(FLT_track[u]['X'][i], FLT_track[u]['Y'][i], 
                                    fov_radius):
                        if not obj.detected:
                            obj.detected = True
                        if u not in obj.detected_by:
                            obj.detected_by.append(u)
        
        # Statistiques
        detected_count = sum(1 for obj in objects if obj.detected)
        objects_per_uav = {}
        for u in range(nUAVs):
            objects_per_uav[u] = sum(1 for obj in objects if u in obj.detected_by)
        
        return {
            'total': len(objects),
            'detected': detected_count,
            'detection_rate': detected_count / len(objects) if len(objects) > 0 else 0,
            'per_uav': objects_per_uav
        }
    
    @staticmethod
    def print_performance_report(metrics: PerformanceMetrics):
        """Afficher le rapport de performance"""
        print("\n" + "="*70)
        print(f"RAPPORT DE PERFORMANCE - {metrics.scenario_name}")
        print("="*70)
        
        print(f"\nNombre d'UAVs: {metrics.num_uavs}")
        
        # Métriques de trajectoire
        if metrics.total_distance:
            print("\n--- TRAJECTOIRES ---")
            for u in range(metrics.num_uavs):
                print(f"UAV {u}:")
                print(f"  Distance totale: {metrics.total_distance.get(u, 0):.1f}m")
                print(f"  M9 L_path: {metrics.total_distance.get(u, 0):.1f}m")
                print(f"  Batterie résiduelle: {metrics.battery_remaining.get(u, 0):.2f}")
                print(f"  Batterie consommée: {metrics.battery_consumed.get(u, 0):.2f}")
            print(f"\nM8a Distance min séparation d_min: {metrics.min_separation_distance:.1f}m")
            print(f"M8b Collisions inter-agents évitées N_coll: {metrics.collisions_avoided}")
        
        # Métriques de temps
        if metrics.total_flight_time:
            print("\n--- TEMPS DE VOL ---")
            for u in range(metrics.num_uavs):
                total = metrics.total_flight_time.get(u, 0)
                glide = metrics.glide_time.get(u, 0)
                soar = metrics.soar_time.get(u, 0)
                engine = metrics.engine_time.get(u, 0)
                
                print(f"UAV {u}:")
                print(f"  Total: {total:.1f}s")
                if total > 0:
                    print(f"  Glide: {glide:.1f}s ({glide/total*100:.1f}%)")
                    print(f"  Soar: {soar:.1f}s ({soar/total*100:.1f}%)")
                    print(f"  Engine: {engine:.1f}s ({engine/total*100:.1f}%)")
                    motor_off = metrics.motor_off_ratio.get(u, (glide + soar) / total * 100)
                    rate = metrics.battery_consumption_rate_ah_per_h.get(u, 0.0)
                    print(f"  M1 Ratio moteur OFF: {motor_off:.1f}%")
                    print(f"  M2 Taux conso batterie: {rate:.3f} Ah/h")
                pe = metrics.potential_energy_j.get(
                    u,
                    {
                        'variation_J': 0.0,
                        'thermal_gain_J': 0.0,
                        'engine_gain_J': 0.0,
                        'gain_J': 0.0,
                        'loss_J': 0.0,
                    },
                )
                print(f"  M3 Variation énergie potentielle: {pe.get('variation_J', 0.0):.1f} J")
                print(f"  M3 Gain thermique cumulé: {pe.get('thermal_gain_J', 0.0):.1f} J")
                print(f"  M3 Gain moteur cumulé: {pe.get('engine_gain_J', 0.0):.1f} J")
                print(f"  M3 Gain total cumulé: {pe.get('gain_J', 0.0):.1f} J")
                print(f"  M3 Perte cumulée énergie potentielle: {pe.get('loss_J', 0.0):.1f} J")
        
        # Métriques thermiques
        if metrics.thermals_generated > 0:
            print("\n--- THERMIQUES ---")
            print(f"Générées: {metrics.thermals_generated}")
            print(f"Détectées: {metrics.thermals_detected} ({metrics.thermals_detected/metrics.thermals_generated*100:.1f}%)")
            print(f"Exploitées: {metrics.thermals_exploited} ({metrics.thermals_exploited/metrics.thermals_generated*100:.1f}%)")
            print(f"M5+ Exploitées uniques: {metrics.thermal_exploited_unique}")
            print(f"M5+ eta_exp,global: {metrics.thermal_exploitation_global_ratio:.1f}%")
            print(f"Rejetées: {metrics.thermals_rejected}")
            
            if metrics.thermals_per_uav:
                print("\nPar UAV:")
                for u, count in metrics.thermals_per_uav.items():
                    detected_u = metrics.thermals_detected_per_uav.get(u, 0)
                    ratio_u = metrics.exploited_detected_ratio.get(u, 0.0)
                    print(f"  UAV {u}: {count} thermiques exploitées | détectées: {detected_u} | M5 exploitées/détectées: {ratio_u:.1f}%")

            if metrics.thermal_exploitation_frequency:
                print("\nM6 Fréquence d'exploitation par thermique:")
                for tid in sorted(metrics.thermal_exploitation_frequency):
                    print(f"  Thermique {tid}: {metrics.thermal_exploitation_frequency[tid]} entrées")

            if metrics.thermal_exploitation_duration_s:
                print("\nM7 Durée d'exploitation par thermique:")
                for tid in sorted(metrics.thermal_exploitation_duration_s):
                    dur = metrics.thermal_exploitation_duration_s[tid]
                    print(f"  Thermique {tid}: {dur:.1f}s")
        
        # Métriques d'endurance
        if metrics.patrol_loops:
            print("\n--- ENDURANCE ---")
            for u in range(metrics.num_uavs):
                loops = metrics.patrol_loops.get(u, 0)
                ratio = metrics.soaring_ratio.get(u, 0.0)
                total = metrics.total_flight_time.get(u, 0)
                glide = metrics.glide_time.get(u, 0)
                soar = metrics.soar_time.get(u, 0)
                engine = metrics.engine_time.get(u, 0)
                bat = metrics.battery_remaining.get(u, 0)
                print(f"UAV {u}:")
                print(f"  Boucles de patrouille: {loops}")
                print(f"  Temps de vol total: {total:.0f}s ({total/60:.1f}min)")
                print(f"  Ratio soaring: {ratio*100:.1f}%")
                print(f"  Répartition: glide {glide:.0f}s | soar {soar:.0f}s | engine {engine:.0f}s")
                print(f"  Batterie restante: {bat:.2f}")
        
        # Métriques de couverture
        if metrics.objects_total > 0:
            print("\n--- COUVERTURE ---")
            print(f"Objets totaux: {metrics.objects_total}")
            print(f"Objets détectés: {metrics.objects_detected} ({metrics.detection_rate*100:.1f}%)")
            print(f"M10 eta_cov: {metrics.detection_rate*100:.1f}%")
            
            if metrics.objects_per_uav:
                print("\nPar UAV:")
                for u, count in metrics.objects_per_uav.items():
                    print(f"  UAV {u}: {count} objets détectés")

        if metrics.algorithm_time_avg_ms > 0:
            print("\n--- CALCUL ---")
            print(f"M11 T_algo moyen: {metrics.algorithm_time_avg_ms:.1f}ms")
            print(f"M11 T_algo max: {metrics.algorithm_time_max_ms:.1f}ms")
        
        print("\n" + "="*70)
    
    @staticmethod
    def save_metrics_to_file(metrics: PerformanceMetrics, filename: str):
        """Sauvegarder les métriques dans un fichier JSON"""
        log_dir = '/home/pix4/GoToWP/log'
        os.makedirs(log_dir, exist_ok=True)
        
        # Construire le chemin complet
        filepath = os.path.join(log_dir, filename)
        
        data = {
            'scenario_name': metrics.scenario_name,
            'num_uavs': metrics.num_uavs,
            'total_distance': metrics.total_distance,
            'path_length': metrics.path_length,
            'collisions_avoided': metrics.collisions_avoided,
            'min_separation_distance': metrics.min_separation_distance,
            'battery_consumed': metrics.battery_consumed,
            'battery_remaining': metrics.battery_remaining,
            'total_flight_time': metrics.total_flight_time,
            'glide_time': metrics.glide_time,
            'soar_time': metrics.soar_time,
            'engine_time': metrics.engine_time,
            'motor_off_ratio': metrics.motor_off_ratio,
            'battery_consumption_rate_ah_per_h': metrics.battery_consumption_rate_ah_per_h,
            'potential_energy_j': metrics.potential_energy_j,
            'thermals_generated': metrics.thermals_generated,
            'thermals_detected': metrics.thermals_detected,
            'thermals_exploited': metrics.thermals_exploited,
            'thermals_rejected': metrics.thermals_rejected,
            'thermals_per_uav': metrics.thermals_per_uav,
            'thermals_detected_per_uav': metrics.thermals_detected_per_uav,
            'exploited_detected_ratio': metrics.exploited_detected_ratio,
            'thermal_exploitation_frequency': metrics.thermal_exploitation_frequency,
            'thermal_exploitation_duration_s': metrics.thermal_exploitation_duration_s,
            'thermal_exploitation_duration_per_uav_s': metrics.thermal_exploitation_duration_per_uav_s,
            'thermal_exploited_unique': metrics.thermal_exploited_unique,
            'thermal_exploitation_global_ratio': metrics.thermal_exploitation_global_ratio,
            'algorithm_time_avg_ms': metrics.algorithm_time_avg_ms,
            'algorithm_time_max_ms': metrics.algorithm_time_max_ms,
            'objects_total': metrics.objects_total,
            'objects_detected': metrics.objects_detected,
            'objects_per_uav': metrics.objects_per_uav,
            'detection_rate': metrics.detection_rate,
            'patrol_loops': metrics.patrol_loops,
            'soaring_ratio': metrics.soaring_ratio
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n✓ Métriques sauvegardées dans {filepath}")

    @staticmethod
    def save_metrics_to_csv(metrics: PerformanceMetrics, filename: str):
        """Sauvegarder les métriques dans un CSV plat dans le dossier log."""
        log_dir = '/home/pix4/GoToWP/log'
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, filename)

        fieldnames = [
            'scenario_name', 'num_uavs', 'uav_id', 'total_distance_m', 'path_length_points',
            'flight_time_s', 'glide_time_s', 'soar_time_s', 'engine_time_s',
            'motor_off_ratio_pct', 'battery_consumed_ah', 'battery_remaining_ah',
            'battery_rate_ah_per_h',
            'potential_energy_variation_j', 'potential_energy_thermal_gain_j',
            'potential_energy_engine_gain_j', 'potential_energy_gain_j', 'potential_energy_loss_j',
            'thermals_detected_uav', 'thermals_exploited_uav', 'exploited_detected_ratio_pct',
            'patrol_loops', 'soaring_ratio_pct',
            'd_min_m', 'n_coll', 'eta_cov_pct', 't_algo_avg_ms', 't_algo_max_ms',
            'eta_exp_global_pct', 'n_exp_unique', 'n_total_thermals'
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for u in range(metrics.num_uavs):
                pe = metrics.potential_energy_j.get(
                    u,
                    {
                        'variation_J': 0.0,
                        'thermal_gain_J': 0.0,
                        'engine_gain_J': 0.0,
                        'gain_J': 0.0,
                        'loss_J': 0.0,
                    },
                )
                writer.writerow({
                    'scenario_name': metrics.scenario_name,
                    'num_uavs': metrics.num_uavs,
                    'uav_id': u,
                    'total_distance_m': metrics.total_distance.get(u, 0.0),
                    'path_length_points': metrics.path_length.get(u, 0),
                    'flight_time_s': metrics.total_flight_time.get(u, 0.0),
                    'glide_time_s': metrics.glide_time.get(u, 0.0),
                    'soar_time_s': metrics.soar_time.get(u, 0.0),
                    'engine_time_s': metrics.engine_time.get(u, 0.0),
                    'motor_off_ratio_pct': metrics.motor_off_ratio.get(u, 0.0),
                    'battery_consumed_ah': metrics.battery_consumed.get(u, 0.0),
                    'battery_remaining_ah': metrics.battery_remaining.get(u, 0.0),
                    'battery_rate_ah_per_h': metrics.battery_consumption_rate_ah_per_h.get(u, 0.0),
                    'potential_energy_variation_j': pe.get('variation_J', 0.0),
                    'potential_energy_thermal_gain_j': pe.get('thermal_gain_J', 0.0),
                    'potential_energy_engine_gain_j': pe.get('engine_gain_J', 0.0),
                    'potential_energy_gain_j': pe.get('gain_J', 0.0),
                    'potential_energy_loss_j': pe.get('loss_J', 0.0),
                    'thermals_detected_uav': metrics.thermals_detected_per_uav.get(u, 0),
                    'thermals_exploited_uav': metrics.thermals_per_uav.get(u, 0),
                    'exploited_detected_ratio_pct': metrics.exploited_detected_ratio.get(u, 0.0),
                    'patrol_loops': metrics.patrol_loops.get(u, 0),
                    'soaring_ratio_pct': metrics.soaring_ratio.get(u, 0.0) * 100.0,
                    'd_min_m': metrics.min_separation_distance,
                    'n_coll': metrics.collisions_avoided,
                    'eta_cov_pct': metrics.detection_rate * 100.0,
                    't_algo_avg_ms': metrics.algorithm_time_avg_ms,
                    't_algo_max_ms': metrics.algorithm_time_max_ms,
                    'eta_exp_global_pct': metrics.thermal_exploitation_global_ratio,
                    'n_exp_unique': metrics.thermal_exploited_unique,
                    'n_total_thermals': metrics.thermals_generated,
                })

        print(f"\n✓ Métriques CSV sauvegardées dans {filepath}")


def select_scenario() -> TestScenario:
    """Interface de sélection de scénario"""
    print("\n" + "="*70)
    print("SÉLECTION DU SCÉNARIO DE TEST")
    print("="*70)
    print("\nScénarios disponibles:")
    print("0 - Préliminaire: Test de collision (100% risque)")
    print("1 - Trajectoire optimale (Propulsé)")
    print("2 - Trajectoire optimale (Planeur)")
    print("3 - Test d'endurance (patrouille + thermiques)")
    print("4 - Test de couverture")
    
    choice = int(input("\nChoisir un scénario (0-4): ") or "0")
    
    if choice == 0:
        return TestScenario.PRELIMINARY_COLLISION
    elif choice == 1:
        return TestScenario.TRAJECTORY_OPTIMAL_POWERED
    elif choice == 2:
        return TestScenario.TRAJECTORY_OPTIMAL_GLIDE
    elif choice == 3:
        return TestScenario.ENDURANCE
    elif choice == 4:
        return TestScenario.COVERAGE
    else:
        print("Scénario invalide, utilisation du scénario préliminaire")
        return TestScenario.PRELIMINARY_COLLISION


if __name__ == "__main__":
    print("="*70)
    print("MODULE DE SCÉNARIOS DE TEST MULTI-UAV")
    print("="*70)
    print("\nCe module fournit les générateurs de scénarios et analyseurs")
    print("pour les tests de performance multi-UAV.")
    print("\nPour exécuter une simulation, utilisez: python3 dronePx4.py")
