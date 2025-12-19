"""
Scénarios de test pour évaluation multi-UAV avec PX4 SITL
Tests d'optimalité, d'endurance et de couverture
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json

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
    powered_time: Dict[int, float] = field(default_factory=dict)
    
    # Métriques thermiques (endurance)
    thermals_generated: int = 0
    thermals_detected: int = 0
    thermals_exploited: int = 0
    thermals_rejected: int = 0
    thermals_per_uav: Dict[int, int] = field(default_factory=dict)
    
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
    
    def is_in_fov(self, uav_x: float, uav_y: float, uav_z: float, fov_radius: float) -> bool:
        """Vérifie si l'objet est dans le champ de vision"""
        distance = np.sqrt((self.x - uav_x)**2 + (self.y - uav_y)**2 + (self.z - uav_z)**2)
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
        
        # Destination commune au centre
        target_x = (params['X_upper_bound'] + params['X_lower_bound']) / 2
        target_y = (params['Y_upper_bound'] + params['Y_lower_bound']) / 2
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
        print( center_obstacle['vertices'])
        print(type(center_obstacle["vertices"]))
        
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
        
        # Destination commune
        target_x = (test_zone['x_max'] + test_zone['x_min']) / 2
        target_y = (test_zone['y_max'] + test_zone['y_min']) / 2
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
        
        # Générer objets de surveillance
        num_objects = np.random.randint(15, 30)
        objects = []
        
        for i in range(num_objects):
            spawn_time = np.random.uniform(0, mission_duration * 0.7)
            lifetime = np.random.uniform(60, 300)  # 1-5 min
            
            obj = SurveillanceObject(
                id=i,
                x=np.random.uniform(params['X_lower_bound'], params['X_upper_bound']),
                y=np.random.uniform(params['Y_lower_bound'], params['Y_upper_bound']),
                z=np.random.uniform(300, 600),
                spawn_time=spawn_time,
                despawn_time=spawn_time + lifetime
            )
            objects.append(obj)
        
        print(f"✓ {num_objects} objets de surveillance générés")
        print(f"✓ Durée de vie: 1-5 min")
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
            powered_time = 0
            
            if len(FLT_track[u]['flight_mode']) > 1:
                for i in range(1, len(FLT_track[u]['flight_mode'])):
                    time_step = FLT_track[u]['flight_time'][i] - FLT_track[u]['flight_time'][i-1]
                    mode = FLT_track[u]['flight_mode'][i]
                    
                    if mode == 'glide':
                        glide_time += time_step
                    elif mode == 'soar':
                        soar_time += time_step
                    elif mode == 'powered':
                        powered_time += time_step
            
            phase_times[u] = {
                'glide': glide_time,
                'soar': soar_time,
                'powered': powered_time,
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
        detected = sum(1 for t in thermals if t.get('detected', False))
        exploited = sum(1 for t in thermals if t.get('exploited', False))
        rejected = sum(1 for t in thermals if t.get('rejected', False))
        
        # Compter par UAV
        thermals_per_uav = {}
        for u in range(nUAVs):
            count = 0
            for t in thermals:
                if u in t.get('exploited_by', []):
                    count += 1
            thermals_per_uav[u] = count
        
        return {
            'detected': detected,
            'exploited': exploited,
            'rejected': rejected,
            'per_uav': thermals_per_uav
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
                                    FLT_track[u]['Z'][i], fov_radius):
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
                print(f"  Batterie résiduelle: {metrics.battery_remaining.get(u, 0):.2f}")
                print(f"  Batterie consommée: {metrics.battery_consumed.get(u, 0):.2f}")
            print(f"\nDistance min séparation: {metrics.min_separation_distance:.1f}m")
            print(f"Collisions évitées: {metrics.collisions_avoided}")
        
        # Métriques de temps
        if metrics.total_flight_time:
            print("\n--- TEMPS DE VOL ---")
            for u in range(metrics.num_uavs):
                total = metrics.total_flight_time.get(u, 0)
                glide = metrics.glide_time.get(u, 0)
                soar = metrics.soar_time.get(u, 0)
                powered = metrics.powered_time.get(u, 0)
                
                print(f"UAV {u}:")
                print(f"  Total: {total:.1f}s")
                if total > 0:
                    print(f"  Glide: {glide:.1f}s ({glide/total*100:.1f}%)")
                    print(f"  Soar: {soar:.1f}s ({soar/total*100:.1f}%)")
                    print(f"  Powered: {powered:.1f}s ({powered/total*100:.1f}%)")
        
        # Métriques thermiques
        if metrics.thermals_generated > 0:
            print("\n--- THERMIQUES ---")
            print(f"Générées: {metrics.thermals_generated}")
            print(f"Détectées: {metrics.thermals_detected} ({metrics.thermals_detected/metrics.thermals_generated*100:.1f}%)")
            print(f"Exploitées: {metrics.thermals_exploited} ({metrics.thermals_exploited/metrics.thermals_generated*100:.1f}%)")
            print(f"Rejetées: {metrics.thermals_rejected}")
            
            if metrics.thermals_per_uav:
                print("\nPar UAV:")
                for u, count in metrics.thermals_per_uav.items():
                    print(f"  UAV {u}: {count} thermiques exploitées")
        
        # Métriques de couverture
        if metrics.objects_total > 0:
            print("\n--- COUVERTURE ---")
            print(f"Objets totaux: {metrics.objects_total}")
            print(f"Objets détectés: {metrics.objects_detected} ({metrics.detection_rate*100:.1f}%)")
            
            if metrics.objects_per_uav:
                print("\nPar UAV:")
                for u, count in metrics.objects_per_uav.items():
                    print(f"  UAV {u}: {count} objets détectés")
        
        print("\n" + "="*70)
    
    @staticmethod
    def save_metrics_to_file(metrics: PerformanceMetrics, filename: str):
        """Sauvegarder les métriques dans un fichier JSON"""
        import os
        
        # Créer le dossier multi_uav_logs s'il n'existe pas
        log_dir = '/home/pix4/GoToWP/multi_uav_logs'
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
            'powered_time': metrics.powered_time,
            'thermals_generated': metrics.thermals_generated,
            'thermals_detected': metrics.thermals_detected,
            'thermals_exploited': metrics.thermals_exploited,
            'thermals_rejected': metrics.thermals_rejected,
            'thermals_per_uav': metrics.thermals_per_uav,
            'objects_total': metrics.objects_total,
            'objects_detected': metrics.objects_detected,
            'objects_per_uav': metrics.objects_per_uav,
            'detection_rate': metrics.detection_rate
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n✓ Métriques sauvegardées dans {filepath}")


def select_scenario() -> TestScenario:
    """Interface de sélection de scénario"""
    print("\n" + "="*70)
    print("SÉLECTION DU SCÉNARIO DE TEST")
    print("="*70)
    print("\nScénarios disponibles:")
    print("0 - Préliminaire: Test de collision (100% risque)")
    print("1 - Trajectoire optimale (Propulsé)")
    print("2 - Trajectoire optimale (Planeur)")
    print("3 - Test d'endurance")
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
