from math import sqrt
import numpy as np
import time
from functools import wraps

from compute import ParetRanking, computePolyArea, sum_normalizedMatrixColumns, wrapN

def benchmark_function(func):
    """Décorateur pour mesurer le temps d'exécution d'une fonction"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

# Version originale de decision_making (sauvegardez votre version originale ici)
def decision_making_original(DM):
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

    return all_ranked_indices # indices de toutes les options du meilleur jusqu'au pire

# Version optimisée
@benchmark_function
def decision_making_optimized(DM):
    """Version optimisée que nous avons proposée"""
    import numpy as np
    
    DM = np.asarray(DM)
    m, n = DM.shape
    
    # Cas simples
    if m == 0:
        return np.array([])
    if m == 1:
        return np.array([0])
    if n == 0:
        return np.arange(m)
    
    # Calculs vectorisés simples
    DM_mean = np.mean(DM, axis=0)
    DM_min = np.min(DM, axis=0)
    
    # D1: Distance à la moyenne
    D1 = -np.sum(np.abs(DM - DM_mean), axis=1)
    
    # D2: Distance au minimum
    D2 = np.sum(np.abs(DM - DM_min), axis=1)
    
    # D3: Périmètre (simplifié)
    indices = np.arange(n)
    D3 = np.zeros(m)
    for i in range(m):
        x = indices
        y = DM[i]
        # Calcul du périmètre
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        D3[i] = np.sum(np.sqrt(dx**2 + dy**2))
    
    # D4: Aire (simplifié)
    D4 = np.zeros(m)
    for i in range(m):
        x = indices
        y = DM[i]
        # Formule du lacet pour l'aire
        area = 0.5 * abs(sum(x[j]*y[(j+1)%n] - x[(j+1)%n]*y[j] for j in range(n)))
        D4[i] = np.sqrt(area)
    
    # D5: Somme des rangs de Pareto par critère
    D5 = np.zeros(m)
    for j in range(n):
        ranks = ParetRanking(DM[:, j].reshape(-1, 1))
        D5 += ranks
    
    # D6: Somme normalisée
    D6 = sum_normalizedMatrixColumns(DM)
    
    # Matrice de décision finale
    newDM = np.column_stack([D1, D2, D3, D4, D5])
    
    # Classement de Pareto
    ranks = ParetRanking(newDM)
    
    # Tri final optimisé
    sorted_indices = np.lexsort((D6, ranks))
    
    return sorted_indices

@benchmark_function 
def decision_making_original_wrapped(DM):
    return decision_making_original(DM)

def generate_test_matrices():
    """Génère différentes matrices de test pour le benchmarking"""
    test_cases = []
    
    # Cas 1: Petite matrice (similaire à votre usage actuel)
    np.random.seed(42)
    small_matrix = np.random.rand(10, 5)
    test_cases.append(("Petite matrice (10x5)", small_matrix))
    
    # Cas 2: Matrice moyenne
    medium_matrix = np.random.rand(50, 5)
    test_cases.append(("Matrice moyenne (50x5)", medium_matrix))
    
    # Cas 3: Grande matrice
    large_matrix = np.random.rand(200, 5)
    test_cases.append(("Grande matrice (200x5)", large_matrix))
    
    # Cas 4: Matrice réaliste - 361 lignes comme dans votre cas d'usage
    np.random.seed(123)  # Pour la reproductibilité
    realistic_matrix = np.random.rand(361, 5)
    # Simuler des valeurs plus réalistes
    realistic_matrix[:, 0] = np.random.choice([-1.0, 0.0], 361)  # C_safety
    realistic_matrix[:, 1] = np.random.uniform(50, 300, 361)     # C_distance
    realistic_matrix[:, 2] = np.random.uniform(0.05, 0.3, 361)  # C_energy
    realistic_matrix[:, 3] = np.random.uniform(0.01, 0.1, 361)  # C_sink
    realistic_matrix[:, 4] = np.random.choice([0, 1, 2, 3], 361) # C_obstacle
    test_cases.append(("Matrice réaliste (361x5)", realistic_matrix))
    
    # Cas 5: Très grande matrice
    very_large_matrix = np.random.rand(500, 5)
    test_cases.append(("Très grande matrice (500x5)", very_large_matrix))
    
    # Cas 6: Matrice avec plus de critères
    wide_matrix = np.random.rand(361, 10)
    test_cases.append(("Matrice large (361x10)", wide_matrix))
    
    return test_cases

def run_benchmark():
    """Execute le benchmark complet avec adaptation pour les grandes matrices"""
    print("=== BENCHMARK DECISION_MAKING ===\n")
    
    test_cases = generate_test_matrices()
    
    for test_name, test_matrix in test_cases:
        print(f"Test: {test_name}")
        print(f"Dimensions: {test_matrix.shape}")
        
        # Adapter le nombre de répétitions selon la taille
        if test_matrix.shape[0] <= 50:
            n_runs = 100
        elif test_matrix.shape[0] <= 200:
            n_runs = 50
        elif test_matrix.shape[0] <= 361:
            n_runs = 20  # Pour la matrice réaliste de 361 lignes
        else:
            n_runs = 10
        
        print(f"Nombre de répétitions: {n_runs}")
        
        # Test version originale
        original_times = []
        print("Testing version originale...", end=" ")
        for i in range(n_runs):
            if i % (n_runs // 4) == 0:
                print(f"{i}/{n_runs}", end=" ")
            try:
                _, exec_time = decision_making_original_wrapped(test_matrix.copy())
                original_times.append(exec_time)
            except Exception as e:
                print(f"\nErreur version originale: {e}")
                original_times = [float('inf')]
                break
        print("Done!")
        
        # Test version optimisée
        optimized_times = []
        print("Testing version optimisée...", end=" ")
        for i in range(n_runs):
            if i % (n_runs // 4) == 0:
                print(f"{i}/{n_runs}", end=" ")
            try:
                _, exec_time = decision_making_optimized(test_matrix.copy())
                optimized_times.append(exec_time)
            except Exception as e:
                print(f"\nErreur version optimisée: {e}")
                optimized_times = [float('inf')]
                break
        print("Done!")
        
        # Calculer les statistiques
        if original_times and optimized_times and original_times[0] != float('inf') and optimized_times[0] != float('inf'):
            avg_original = np.mean(original_times) * 1000  # en ms
            avg_optimized = np.mean(optimized_times) * 1000  # en ms
            std_original = np.std(original_times) * 1000
            std_optimized = np.std(optimized_times) * 1000
            min_original = np.min(original_times) * 1000
            min_optimized = np.min(optimized_times) * 1000
            max_original = np.max(original_times) * 1000
            max_optimized = np.max(optimized_times) * 1000
            
            speedup = avg_original / avg_optimized
            improvement = ((avg_original - avg_optimized) / avg_original) * 100
            
            print(f"Version originale:")
            print(f"  Moyenne: {avg_original:.3f} ± {std_original:.3f} ms")
            print(f"  Min/Max: {min_original:.3f} / {max_original:.3f} ms")
            print(f"Version optimisée:")
            print(f"  Moyenne: {avg_optimized:.3f} ± {std_optimized:.3f} ms")
            print(f"  Min/Max: {min_optimized:.3f} / {max_optimized:.3f} ms")
            print(f"Accélération: {speedup:.2f}x")
            print(f"Amélioration: {improvement:.1f}%")
            
            if speedup > 1.1:  # Amélioration significative
                print("✅ Version optimisée significativement plus rapide")
            elif speedup > 1.0:
                print("🟡 Version optimisée légèrement plus rapide")
            elif speedup > 0.9:
                print("🟡 Performances similaires")
            else:
                print("❌ Version optimisée plus lente")
        else:
            print("⚠️  Erreur dans l'exécution")
        
        print("-" * 60)

def quick_realistic_test():
    """Test rapide avec seulement la matrice réaliste de 361 lignes"""
    print("=== TEST RAPIDE MATRICE RÉALISTE (361x5) ===\n")
    
    # Générer une matrice similaire à vos données réelles
    np.random.seed(42)
    realistic_matrix = np.zeros((361, 5))
    
    # Simuler vos critères réels
    realistic_matrix[:, 0] = np.random.choice([-1.0, 0.0], 361, p=[0.3, 0.7])  # C_safety
    realistic_matrix[:, 1] = np.random.uniform(50, 500, 361)                    # C_distance
    realistic_matrix[:, 2] = np.random.uniform(0.05, 0.5, 361)                 # C_energy
    realistic_matrix[:, 3] = np.random.uniform(0.01, 0.15, 361)                # C_sink
    realistic_matrix[:, 4] = np.random.choice([0, -1], 361, p=[0.8, 0.2])      # C_obstacle
    
    print(f"Matrice de test: {realistic_matrix.shape}")
    print(f"Échantillon des données:")
    print(realistic_matrix[:5])
    print("...")
    
    n_runs = 10
    print(f"Nombre de tests: {n_runs}\n")
    
    # Test version originale
    print("Test version originale...")
    original_times = []
    for i in range(n_runs):
        start_time = time.perf_counter()
        result_original = decision_making_original(realistic_matrix.copy())
        end_time = time.perf_counter()
        original_times.append(end_time - start_time)
        print(f"  Run {i+1}: {(end_time - start_time)*1000:.2f} ms")
    
    # Test version optimisée
    print("\nTest version optimisée...")
    optimized_times = []
    for i in range(n_runs):
        start_time = time.perf_counter()
        result_optimized, _ = decision_making_optimized(realistic_matrix.copy())
        end_time = time.perf_counter()
        optimized_times.append(end_time - start_time)
        print(f"  Run {i+1}: {(end_time - start_time)*1000:.2f} ms")
    
    # Statistiques
    avg_original = np.mean(original_times) * 1000
    avg_optimized = np.mean(optimized_times) * 1000
    
    print(f"\n📊 RÉSULTATS:")
    print(f"Version originale: {avg_original:.2f} ms (moyenne)")
    print(f"Version optimisée: {avg_optimized:.2f} ms (moyenne)")
    print(f"Différence: {avg_original - avg_optimized:.2f} ms")
    print(f"Accélération: {avg_original / avg_optimized:.2f}x")
    
    # Vérifier la cohérence
    print(f"\n🔍 VÉRIFICATION COHÉRENCE:")
    print(f"Résultats identiques: {np.array_equal(result_original, result_optimized)}")
    if not np.array_equal(result_original, result_optimized):
        print(f"Premier différent à l'index: {np.where(result_original != result_optimized)[0][0] if len(np.where(result_original != result_optimized)[0]) > 0 else 'Aucun'}")

def generate_multi_drone_test_matrices():
    """Génère des matrices de test pour différents scénarios multi-drones"""
    test_cases = []
    
    # Scénario 1: 1 drone (cas de base)
    np.random.seed(42)
    single_drone_matrix = generate_realistic_matrix(361, 5)
    test_cases.append(("1 drone (361x5)", single_drone_matrix, 1))
    
    # Scénario 2: 2 drones
    two_drone_matrix = generate_realistic_matrix(722, 5)  # 361 * 2
    test_cases.append(("2 drones (722x5)", two_drone_matrix, 2))
    
    # Scénario 3: 3 drones
    three_drone_matrix = generate_realistic_matrix(1083, 5)  # 361 * 3
    test_cases.append(("3 drones (1083x5)", three_drone_matrix, 3))
    
    # Scénario 4: 5 drones
    five_drone_matrix = generate_realistic_matrix(1805, 5)  # 361 * 5
    test_cases.append(("5 drones (1805x5)", five_drone_matrix, 5))
    
    return test_cases

def generate_realistic_matrix(rows, cols):
    """Génère une matrice avec des valeurs réalistes pour le contexte UAV"""
    matrix = np.zeros((rows, cols))
    
    # C_safety: -1 (collision), 0 (sécuritaire)
    matrix[:, 0] = np.random.choice([-1.0, 0.0], rows, p=[0.2, 0.8])
    
    # C_distance: distance au waypoint (50-500m)
    matrix[:, 1] = np.random.uniform(50, 500, rows)
    
    # C_energy: consommation d'énergie normalisée (0.05-0.5)
    matrix[:, 2] = np.random.uniform(0.05, 0.5, rows)
    
    # C_sink: taux de descente normalisé (0.01-0.15)
    matrix[:, 3] = np.random.uniform(0.01, 0.15, rows)
    
    # C_obstacle: collision avec obstacle (-1) ou pas (0)
    matrix[:, 4] = np.random.choice([0, -1], rows, p=[0.85, 0.15])
    
    return matrix

def run_multi_drone_benchmark():
    """Execute le benchmark pour différents nombres de drones"""
    print("=== BENCHMARK MULTI-DRONES DECISION_MAKING ===\n")
    
    test_cases = generate_multi_drone_test_matrices()
    results = []
    
    for test_name, test_matrix, num_drones in test_cases:
        print(f"Test: {test_name}")
        print(f"Dimensions: {test_matrix.shape}")
        print(f"Candidats par drone: {test_matrix.shape[0] // num_drones}")
        
        # Adapter le nombre de répétitions selon la taille
        if test_matrix.shape[0] <= 1000:
            n_runs = 20
        elif test_matrix.shape[0] <= 3000:
            n_runs = 10
        else:
            n_runs = 5
        
        print(f"Nombre de répétitions: {n_runs}")
        
        # Test version originale
        original_times = []
        print("Testing version originale...", end=" ")
        for i in range(n_runs):
            if n_runs >= 10 and i % (n_runs // 4) == 0:
                print(f"{i}/{n_runs}", end=" ")
            try:
                _, exec_time = decision_making_original_wrapped(test_matrix.copy())
                original_times.append(exec_time)
            except Exception as e:
                print(f"\nErreur version originale: {e}")
                original_times = [float('inf')]
                break
        print("Done!")
        
        # Test version optimisée
        optimized_times = []
        print("Testing version optimisée...", end=" ")
        for i in range(n_runs):
            if n_runs >= 10 and i % (n_runs // 4) == 0:
                print(f"{i}/{n_runs}", end=" ")
            try:
                _, exec_time = decision_making_optimized(test_matrix.copy())
                optimized_times.append(exec_time)
            except Exception as e:
                print(f"\nErreur version optimisée: {e}")
                optimized_times = [float('inf')]
                break
        print("Done!")
        
        # Calculer les statistiques
        if original_times and optimized_times and original_times[0] != float('inf') and optimized_times[0] != float('inf'):
            avg_original = np.mean(original_times) * 1000  # en ms
            avg_optimized = np.mean(optimized_times) * 1000  # en ms
            std_original = np.std(original_times) * 1000
            std_optimized = np.std(optimized_times) * 1000
            
            speedup = avg_original / avg_optimized
            improvement = ((avg_original - avg_optimized) / avg_original) * 100
            
            # Sauvegarder les résultats pour le résumé
            results.append({
                'drones': num_drones,
                'rows': test_matrix.shape[0],
                'original': avg_original,
                'optimized': avg_optimized,
                'speedup': speedup,
                'improvement': improvement
            })
            
            print(f"Version originale: {avg_original:.2f} ± {std_original:.2f} ms")
            print(f"Version optimisée: {avg_optimized:.2f} ± {std_optimized:.2f} ms")
            print(f"Accélération: {speedup:.2f}x")
            print(f"Amélioration: {improvement:.1f}%")
            
            if speedup > 1.1:
                print("✅ Version optimisée significativement plus rapide")
            elif speedup > 1.0:
                print("🟡 Version optimisée légèrement plus rapide")
            else:
                print("❌ Version optimisée scale moins bien")
        else:
            print("⚠️  Erreur dans l'exécution")
        
        print("-" * 60)
    
    # Résumé des résultats
    print_benchmark_summary(results)

def print_benchmark_summary(results):
    """Affiche un résumé des résultats de benchmark"""
    print("\n=== RÉSUMÉ DES PERFORMANCES ===\n")
    
    print(f"{'Drones':<8} {'Candidats':<10} {'Original (ms)':<15} {'Optimisé (ms)':<15} {'Accélération':<12} {'Amélioration':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['drones']:<8} {result['rows']:<10} {result['original']:<15.2f} "
              f"{result['optimized']:<15.2f} {result['speedup']:<12.2f}x {result['improvement']:<12.1f}%")
    
    # Moyennes
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])
        print("-" * 80)
        print(f"{'MOYENNE':<8} {'':<10} {'':<15} {'':<15} {avg_speedup:<12.2f}x {avg_improvement:<12.1f}%")
    
    # Analyse de scalabilité
    print("\n=== ANALYSE DE SCALABILITÉ ===")
    if len(results) > 1:
        print(f"Scalabilité (1 → {results[-1]['drones']} drones):")
        print(f"  Original: {results[0]['original']:.2f} ms → {results[-1]['original']:.2f} ms")
        print(f"  Optimisé: {results[0]['optimized']:.2f} ms → {results[-1]['optimized']:.2f} ms")
        
        original_ratio = results[-1]['original'] / results[0]['original']
        optimized_ratio = results[-1]['optimized'] / results[0]['optimized']
        
        print(f"  Facteur d'augmentation original: {original_ratio:.1f}x")
        print(f"  Facteur d'augmentation optimisé: {optimized_ratio:.1f}x")
        
        if optimized_ratio < original_ratio:
            print("✅ Version optimisée scale mieux avec le nombre de drones")
        else:
            print("❌ Version optimisée scale moins bien")

def stress_test_large_fleet():
    """Test de stress avec une grande flotte de drones"""
    print("\n=== TEST DE STRESS GRANDE FLOTTE ===\n")
    
    # Test avec 50 drones (18050 candidats)
    large_fleet_sizes = [30, 50, 100]
    
    for num_drones in large_fleet_sizes:
        rows = 361 * num_drones
        print(f"Test stress: {num_drones} drones ({rows} candidats)")
        
        # Générer matrice de test
        test_matrix = generate_realistic_matrix(rows, 5)
        
        # Test unique pour chaque version (pas de répétitions multiples)
        print("Test version originale...", end=" ")
        try:
            start_time = time.perf_counter()
            result_original = decision_making_original(test_matrix.copy())
            original_time = (time.perf_counter() - start_time) * 1000
            print(f"Done! ({original_time:.0f} ms)")
        except Exception as e:
            print(f"Failed: {e}")
            original_time = float('inf')
        
        print("Test version optimisée...", end=" ")
        try:
            start_time = time.perf_counter()
            result_optimized, _ = decision_making_optimized(test_matrix.copy())
            optimized_time = (time.perf_counter() - start_time) * 1000
            print(f"Done! ({optimized_time:.0f} ms)")
        except Exception as e:
            print(f"Failed: {e}")
            optimized_time = float('inf')
        
        if original_time != float('inf') and optimized_time != float('inf'):
            speedup = original_time / optimized_time
            print(f"Résultats: Original {original_time:.0f} ms vs Optimisé {optimized_time:.0f} ms")
            print(f"Accélération: {speedup:.2f}x")
            
            # Vérifier la cohérence pour les petites tailles seulement
            if num_drones <= 30:
                consistent = np.array_equal(result_original, result_optimized)
                print(f"Cohérence: {'✅' if consistent else '❌'}")
        else:
            print("Test échoué")
        
        print("-" * 50)

def compare_scaling_behavior():
    """Compare le comportement de montée en charge des deux versions"""
    print("\n=== COMPARAISON MONTÉE EN CHARGE ===\n")
    
    drone_counts = [1, 2, 5, 10, 15, 20]
    original_times = []
    optimized_times = []
    
    for num_drones in drone_counts:
        rows = 361 * num_drones
        test_matrix = generate_realistic_matrix(rows, 5)
        
        print(f"Test {num_drones} drones ({rows} candidats)...", end=" ")
        
        # Test version originale (3 runs pour moyenne)
        times_orig = []
        for _ in range(3):
            start = time.perf_counter()
            decision_making_original(test_matrix.copy())
            times_orig.append(time.perf_counter() - start)
        avg_orig = np.mean(times_orig) * 1000
        
        # Test version optimisée (3 runs pour moyenne)
        times_opt = []
        for _ in range(3):
            start = time.perf_counter()
            result, _ = decision_making_optimized(test_matrix.copy())
            times_opt.append(time.perf_counter() - start)
        avg_opt = np.mean(times_opt) * 1000
        
        original_times.append(avg_orig)
        optimized_times.append(avg_opt)
        
        print(f"Orig: {avg_orig:.1f}ms, Opt: {avg_opt:.1f}ms, Speedup: {avg_orig/avg_opt:.1f}x")
    
    # Affichage du tableau de scaling
    print(f"\n{'Drones':<8} {'Original (ms)':<15} {'Optimisé (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    for i, num_drones in enumerate(drone_counts):
        speedup = original_times[i] / optimized_times[i]
        print(f"{num_drones:<8} {original_times[i]:<15.1f} {optimized_times[i]:<15.1f} {speedup:<10.1f}x")

# Mise à jour de la fonction main
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_realistic_test()
        elif sys.argv[1] == "multi":
            run_multi_drone_benchmark()
        elif sys.argv[1] == "stress":
            stress_test_large_fleet()
        elif sys.argv[1] == "scaling":
            compare_scaling_behavior()
        elif sys.argv[1] == "all":
            print("Exécution de tous les tests...\n")
            quick_realistic_test()
            print("\n" + "="*80 + "\n")
            run_multi_drone_benchmark()
            print("\n" + "="*80 + "\n")
            stress_test_large_fleet()
            print("\n" + "="*80 + "\n")
            compare_scaling_behavior()
    else:
        # Exécuter le benchmark complet original
        print("Exécution de tous les tests...\n")
        quick_realistic_test()
        print("\n" + "="*80 + "\n")
        run_multi_drone_benchmark()
        print("\n" + "="*80 + "\n")
        stress_test_large_fleet()
        print("\n" + "="*80 + "\n")
        compare_scaling_behavior()