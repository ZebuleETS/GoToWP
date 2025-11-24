# Intégration Simulation Planeur avec PX4 SITL

Guide complet pour intégrer votre simulation Python de planeur avec thermiques dans PX4 SITL + Gazebo.

## 📋 Prérequis

### Système
- Ubuntu 20.04 / 22.04 (ou WSL2 sur Windows)
- Python 3.8+
- Au moins 4 Go RAM
- 20 Go d'espace disque

### Installation PX4

```bash
# 1. Cloner PX4-Autopilot
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot

# 2. Installer les dépendances
bash ./Tools/setup/ubuntu.sh

# 3. Redémarrer le terminal ou recharger
source ~/.bashrc

# 4. Tester l'installation
make px4_sitl gazebo_plane
```

### Installation Python

```bash
# Installer les dépendances Python
pip install mavsdk numpy pymavlink

# Vérifier vos modules existants
# GoToWP.py, compute.py, trajectory.py, thermal.py doivent être accessibles
```

## 🚀 Démarrage Rapide

### 1. Structure des Fichiers

```
votre_projet/
├── example.py                  # Votre simulation originale
├── example_px4_integrated.py   # Version intégrée PX4 (fournie)
├── px4_config.py              # Configuration paramètres PX4
├── launch_simulation.sh       # Script de lancement
├── GoToWP.py                  # Vos modules
├── compute.py
├── trajectory.py
└── thermal.py
```

### 2. Lancer la Simulation

**Option A: Script automatique**
```bash
chmod +x launch_simulation.sh
./launch_simulation.sh
```

**Option B: Manuel**

Terminal 1 - PX4 SITL:
```bash
cd ~/PX4-Autopilot
make px4_sitl gazebo_plane
```

Terminal 2 - Votre simulation:
```bash
cd votre_projet
source ./Mav/bin/activate
python3 dronePx4.py
```

Terminal 3 - QGroundControl (optionnel):
```bash
# Télécharger depuis: http://qgroundcontrol.com/
./QGroundControl.AppImage
```

## 🔧 Configuration

### Paramètres PX4

Avant la première utilisation, configurez les paramètres:

```bash
python3 px4_config.py
```

Ou manuellement via QGroundControl:
1. Connecter à UDP sur port 14550
2. Aller dans **Settings → Parameters**
3. Modifier les paramètres Fixed Wing (FW_*)

### Paramètres Importants

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `FW_AIRSPD_MIN` | 8.0 m/s | Vitesse minimale (stall) |
| `FW_AIRSPD_MAX` | 30.0 m/s | Vitesse maximale |
| `FW_AIRSPD_TRIM` | 18.0 m/s | Vitesse de croisière |
| `NAV_LOITER_RAD` | 80.0 m | Rayon de loiter |
| `NAV_ACC_RAD` | 20.0 m | Rayon acceptance waypoint |

## 🎮 Architecture de l'Intégration

### Flux de Données

```
┌─────────────────────┐
│  Votre Simulation   │
│  (Python)           │
│  - Path Planning    │
│  - Thermal Detection│
│  - Energy Mgmt      │
└──────────┬──────────┘
           │
           │ XYZ Coordinates
           ▼
┌─────────────────────┐
│  PX4SITLBridge      │
│  - XYZ → NED        │
│  - NED → GPS        │
│  - Command Sending  │
└──────────┬──────────┘
           │
           │ MAVLink/MAVSDK
           ▼
┌─────────────────────┐
│  PX4 SITL           │
│  - Flight Control   │
│  - State Estimation │
│  - Dynamics         │
└──────────┬──────────┘
           │
           │ Gazebo Protocol
           ▼
┌─────────────────────┐
│  Gazebo             │
│  - 3D Visualization │
│  - Physics Engine   │
│  - Sensors          │
└─────────────────────┘
```

### Systèmes de Coordonnées

**Votre Simulation (XYZ)**
- X: Est (m)
- Y: Nord (m)
- Z: Altitude AGL (m)
- Origine: Arbitraire dans votre espace

**PX4 (NED)**
- N: Nord (m)
- E: Est (m)
- D: Bas (m, négatif vers le haut)
- Origine: Position home du drone

**Conversion**
```python
# XYZ → NED
north = y
east = x
down = -(z - working_floor)

# NED → GPS
lat = home_lat + north / 111320.0
lon = home_lon + east / (111320.0 * cos(home_lat))
alt = home_alt - down
```

## 📊 Modes de Vol

Votre simulation supporte différents modes:

### 1. Mode Glide (Plané)
- Navigation directe vers waypoints
- Consommation batterie normale
- Pas de gain d'altitude

### 2. Mode Soaring (Spirale en thermique)
- Détection automatique de thermiques
- Gain d'altitude dans les ascendances
- Économie d'énergie

### 3. Mode Evaluation
- Exploration du centre de la thermique
- Mesure de l'intensité
- Décision d'exploitation

## 🛠️ Personnalisation

### Modifier les Paramètres du Planeur

Dans `example_px4_integrated.py`, modifiez `UAV_data`:

```python
UAV_data = dict()
UAV_data['maximum_battery_capacity'] = 10.0  # Ah
UAV_data['empty_weight'] = 1.6              # kg
UAV_data['wing_area'] = 0.5                 # m²
UAV_data['wing_aspect_ratio'] = 15.7        # sans unité
UAV_data['max_airspeed'] = 30.0             # m/s
UAV_data['min_airspeed'] = 8.0              # m/s
```

### Modifier la Zone de Simulation

```python
params = dict()
params['working_floor'] = 600.0           # Altitude de vol (m)
params['X_lower_bound'] = 0.0             # Limite X min (m)
params['X_upper_bound'] = 6000.0          # Limite X max (m)
params['Y_lower_bound'] = 0.0             # Limite Y min (m)
params['Y_upper_bound'] = 6000.0          # Limite Y max (m)
```

### Ajouter des Obstacles

```python
obstacles = [
    {
        'x': 1000,        # Position X (m)
        'y': 1500,        # Position Y (m)
        'radius': 200,    # Rayon (m)
        'z_min': 0,       # Alt min (m)
        'z_max': 800      # Alt max (m)
    }
]
```

## 🐛 Débogage

### Problèmes Communs

**PX4 ne démarre pas**
```bash
# Nettoyer et rebuild
cd ~/PX4-Autopilot
make clean
make px4_sitl gazebo_plane
```

**Connexion MAVSDK échoue**
- Vérifier que PX4 écoute sur port 14540: `netstat -ln | grep 14540`
- Attendre 10-15s après le démarrage de PX4
- Vérifier les logs: `~/PX4-Autopilot/build/px4_sitl_default/logs/`

**Drone ne décolle pas**
- Vérifier GPS fix dans QGroundControl
- Vérifier que les paramètres FW_* sont corrects
- Essayer mode manuel d'abord

**Coordonnées incorrectes**
- Vérifier la conversion XYZ→NED dans `PX4SITLBridge`
- Vérifier `working_floor` correspond à l'altitude de simulation
- Afficher les coordonnées intermédiaires pour debug

### Logs Utiles

```bash
# Logs PX4
tail -f ~/PX4-Autopilot/build/px4_sitl_default/logs/*.ulg

# Logs Python
python3 example_px4_integrated.py 2>&1 | tee simulation.log

# Logs Gazebo
tail -f ~/.gazebo/server-*.log
```

### Mode Debug

Activez le mode verbose dans votre script:

```python
# Afficher toutes les positions
print(f"XYZ: ({x:.1f}, {y:.1f}, {z:.1f})")
print(f"NED: ({n:.1f}, {e:.1f}, {d:.1f})")
print(f"GPS: ({lat:.6f}, {lon:.6f}, {alt:.1f})")
```

## 📈 Monitoring

### Via QGroundControl
1. Vue 3D du drone en temps réel
2. Graphiques de télémétrie
3. État de la batterie
4. Position sur carte

### Via Console Python
```python
# Ajouter dans la boucle principale
if iteration % 10 == 0:
    print(f"Alt: {FLT_track[0]['Z'][-1]:.1f}m | "
          f"Battery: {FLT_track[0]['battery_capacity'][-1]:.2f} | "
          f"Mode: {FLT_track[0]['flight_mode'][-1]}")
```

### Via MAVLink Inspector
```bash
# Installer mavlink-routerd
sudo apt install mavlink-router

# Monitorer les messages
mavlink-router -e 127.0.0.1:14550 127.0.0.1:14540
```

## 🎯 Prochaines Étapes

### Améliorations Possibles

1. **Visualisation des Thermiques dans Gazebo**
   - Ajouter des plugins Gazebo pour visualiser les thermiques
   - Créer des marqueurs visuels

2. **Multi-UAV**
   - Étendre pour supporter plusieurs drones
   - Coordination entre drones

3. **Intégration Capteurs**
   - Utiliser les capteurs simulés Gazebo
   - Simuler variomètre, GPS drift

4. **Machine Learning**
   - Apprendre les stratégies de vol en thermique
   - Optimisation de trajectoire avec RL

5. **Hardware-in-the-Loop (HITL)**
   - Connecter un vrai autopilote
   - Tester sur matériel réel

## 📚 Ressources

- [Documentation PX4](https://docs.px4.io/)
- [MAVSDK Python](https://mavsdk.mavlink.io/main/en/python/)
- [MAVLink Protocol](https://mavlink.io/)
- [Gazebo Simulator](http://gazebosim.org/)
- [QGroundControl](http://qgroundcontrol.com/)

## 🤝 Support

En cas de problème:
1. Consulter les logs
2. Vérifier les paramètres PX4
3. Tester avec une mission simple d'abord
4. Forum PX4: https://discuss.px4.io/

## 📝 Licence

Votre code + intégration PX4 (BSD-3-Clause comme PX4)


# Intégration Multi-UAV Simulation Planeur avec PX4 SITL

Guide complet pour intégrer votre simulation Python multi-UAV de planeur avec thermiques dans PX4 SITL + Gazebo.

## 📋 Architecture Multi-UAV

### Système de Ports MAVLink

Chaque UAV utilise des ports différents:

| UAV ID | MAVLink UDP | MAVLink TCP | Simulation |
|--------|-------------|-------------|------------|
| 0      | 14540       | 4560        | 14560      |
| 1      | 14541       | 4561        | 14561      |
| 2      | 14542       | 4562        | 14562      |
| ...    | ...         | ...         | ...        |
| N      | 14540+N     | 4560+N      | 14560+N    |

### Spawn des Drones

Les drones sont automatiquement espacés pour éviter les collisions:
- Espacement horizontal: 200m entre chaque UAV
- Disposition en grille 3×N
- UAV 0: (500, 500), UAV 1: (700, 500), UAV 2: (900, 500)
- UAV 3: (500, 700), UAV 4: (700, 700), etc.

## 🚀 Démarrage Rapide Multi-UAV

### Option 1: Script Automatique (Recommandé)

```bash
# Rendre le script exécutable
chmod +x launch_multi_uav.sh

# Lancer avec 3 UAVs (par défaut)
./launch_multi_uav.sh

# Ou spécifier le nombre d'UAVs (max 10)
./launch_multi_uav.sh 5
```

Le script va:
1. ✓ Nettoyer les processus existants
2. ✓ Lancer Gazebo
3. ✓ Lancer N instances PX4
4. ✓ Vérifier que tout fonctionne
5. ✓ Attendre votre confirmation
6. ✓ Lancer votre simulation Python

### Option 2: Lancement Manuel

**Terminal 1 - Gazebo:**
```bash
cd ~/PX4-Autopilot
DONT_RUN=1 make px4_sitl_default gazebo
```

**Terminal 2, 3, 4... - Instances PX4:**
```bash
# UAV 0
cd ~/PX4-Autopilot
PX4_INSTANCE=0 PX4_SIM_MODEL=plane ./build/px4_sitl_default/bin/px4 -i 0

# UAV 1 (nouveau terminal)
PX4_INSTANCE=1 PX4_SIM_MODEL=plane ./build/px4_sitl_default/bin/px4 -i 1

# UAV 2 (nouveau terminal)
PX4_INSTANCE=2 PX4_SIM_MODEL=plane ./build/px4_sitl_default/bin/px4 -i 2

# etc...
```

**Terminal Final - Simulation:**
```bash
python3 example_px4_integrated.py
# Entrez le nombre d'UAVs quand demandé
```

## 🎮 Contrôle et Monitoring

### QGroundControl Multi-UAV

1. Lancer QGroundControl
2. Se connecter sur `udp://:14550`
3. Utiliser le sélecteur de véhicule (en haut) pour changer entre UAVs
4. Chaque UAV apparaît comme un véhicule séparé

### Monitoring en Temps Réel

La simulation affiche l'état de tous les UAVs:

```
======================================================================
[t=150.0s | iter=150] Temps décision: 12.34ms
======================================================================
  UAV 0: (2345, 1823, 720) | soaring | Bat: 8.4 | Therm: 0
  UAV 1: (3200, 2100, 650) | glide | Bat: 9.1
  UAV 2: (1890, 1650, 580) | evaluation | Bat: 8.8 | Therm: 2
```

### Logs Multi-UAV

Les logs sont automatiquement séparés:
```
multi_uav_logs/
├── gazebo.log
├── px4_uav_0.log
├── px4_uav_1.log
├── px4_uav_2.log
└── ...
```

## 🔧 Configuration Multi-UAV

### Paramètres Clés

Dans `example_px4_integrated.py`:

```python
# Nombre d'UAVs
nUAVs = 3  # 1 à 10

# Espacement au spawn
spacing = 200  # mètres entre UAVs

# Nombre de thermiques (généralement ≥ nUAVs)
num_thermals = max(3, nUAVs)
```

### Allocation des Thermiques

Chaque UAV cible une thermique différente:
```python
# UAV 0 → Thermique 0
# UAV 1 → Thermique 1
# UAV 2 → Thermique 2
# UAV 3 → Thermique 0 (cycle)
target_thermal_idx = u % len(active_thermals)
```

### Évitement de Collision

L'évitement est géré par:
1. **Espacement initial**: 200m minimum entre UAVs
2. **Safe distance**: 30m (paramètre `params['safe_distance']`)
3. **Navigation indépendante**: Chaque UAV a sa propre trajectoire

## 📊 Fonctionnalités Multi-UAV

### 1. Thermiques Partagées

Plusieurs UAVs peuvent:
- ✓ Détecter la même thermique
- ✓ L'évaluer indépendamment
- ✓ L'exploiter simultanément (spirales à altitudes différentes)

### 2. Synchronisation

Le contrôleur `MultiUAVController` gère:
```python
# Connexion parallèle de tous les UAVs
await controller.initialize_all_uavs()

# Décollage simultané
await controller.arm_and_takeoff_all()

# Mise à jour synchronisée à 10Hz
await controller.update_all_from_simulation(FLT_track)

# Atterrissage coordonné
await controller.land_all()
```

### 3. Gestion d'Erreur

Si un UAV échoue:
- Les autres continuent
- Logs séparés pour diagnostic
- Possibilité de RTL individuel

## 🧪 Scénarios de Test

### Scénario 1: Formation de Base (3 UAVs)
```bash
./launch_multi_uav.sh 3
```
- 3 UAVs, 3 thermiques
- Test basique de coordination

### Scénario 2: Essaim (5-7 UAVs)
```bash
./launch_multi_uav.sh 6
```
- Comportement d'essaim
- Partage de thermiques
- Test de scalabilité

### Scénario 3: Stress Test (10 UAVs)
```bash
./launch_multi_uav.sh 10
```
- Limite système
- Performance de décision
- Gestion des ressources

## ⚡ Optimisations Performance

### 1. Calculs Parallèles

Utilise `asyncio.gather()` pour:
- Connexions simultanées
- Décollages parallèles
- Mises à jour synchronisées

### 2. Fréquence de Mise à Jour

```python
self.update_rate = 10  # Hz par défaut

# Pour plus d'UAVs, réduire:
if nUAVs > 5:
    self.update_rate = 5  # 5Hz pour >5 UAVs
```

### 3. Mémoire

Estimation mémoire:
- 1 UAV: ~200 MB
- 3 UAVs: ~500 MB
- 10 UAVs: ~1.5 GB

Minimum recommandé: 4 GB RAM

## 🐛 Débogage Multi-UAV

### Problèmes Courants

**Problème: UAVs ne se connectent pas**
```bash
# Vérifier les ports
netstat -ln | grep 1454

# Devrait montrer:
# 14540, 14541, 14542, etc.
```

**Problème: Collisions au spawn**
```python
# Augmenter l'espacement
spacing = 300  # au lieu de 200
```

**Problème: Performance dégradée**
```bash
# Voir les temps de décision
# Si >50ms avec 5+ UAVs, réduire update_rate
```

### Logs de Débogage

```python
# Activer logs verbeux pour un UAV
print(f"[UAV {u}] Position: {FLT_track[u]['X'][-1]:.1f}")
print(f"[UAV {u}] Mode: {FLT_track[u]['flight_mode'][-1]}")
print(f"[UAV {u}] Thermal: {FLT_track[u]['current_thermal_id']}")
```

### Vérification Ports

```bash
# Script de vérification
for i in {0..9}; do
    PORT=$((14540 + i))
    echo "UAV $i:"
    netstat -ln | grep $PORT || echo "  Port $PORT libre"
done
```

## 📈 Métriques de Simulation

### Statistiques Collectées

Pour chaque UAV:
- Position finale
- Batterie consommée
- Temps de vol
- Distribution des modes (glide/soaring/evaluation)
- Thermiques exploitées

### Analyse Post-Simulation

```python
# Après la simulation
total_battery_used = sum(
    UAV_data['maximum_battery_capacity'] - FLT_track[u]['battery_capacity'][-1]
    for u in range(nUAVs)
)

total_soaring_time = sum(
    FLT_track[u]['flight_mode'].count('soaring')
    for u in range(nUAVs)
)

print(f"Batterie totale utilisée: {total_battery_used:.2f} Ah")
print(f"Temps de soaring total: {total_soaring_time}s")
```

## 🎯 Cas d'Usage Avancés

### 1. Mission de Surveillance

```python
# Définir zones de surveillance pour chaque UAV
surveillance_zones = [
    {'x': 1000, 'y': 1000, 'radius': 500},  # UAV 0
    {'x': 3000, 'y': 1000, 'radius': 500},  # UAV 1
    {'x': 2000, 'y': 3000, 'radius': 500},  # UAV 2
]
```

### 2. Relais de Communication

```python
# UAVs maintiennent connectivité
max_distance_between_uavs = 1000  # mètres
# Ajuster trajectoires pour maintenir maillage
```

### 3. Recherche Collaborative

```python
# Partager détections de thermiques entre UAVs
shared_thermal_map = ThermalMap()
# Tous les UAVs ajoutent leurs détections
```

## 🔬 Expérimentations

### Test 1: Scalabilité
Objectif: Mesurer temps de calcul vs nombre d'UAVs

```bash
for n in 1 2 3 5 7 10; do
    echo "Test avec $n UAVs"
    ./launch_multi_uav.sh $n
    # Noter temps de décision moyen
done
```

### Test 2: Efficacité Énergétique
Objectif: Comparer consommation solo vs groupe

```python
# Solo: 1 UAV avec 3 thermiques
# Groupe: 3 UAVs avec 3 thermiques
# Mesurer batterie/distance parcourue
```

### Test 3: Partage d'Information
Objectif: Avantage de la communication inter-UAV

```python
# Scénario A: UAVs indépendants
# Scénario B: UAVs partagent détections thermiques
# Comparer temps de mission et énergie
```

## 📚 Ressources

### Documentation
- [PX4 Multi-Vehicle Simulation](https://docs.px4.io/main/en/simulation/multi-vehicle-simulation.html)
- [MAVSDK Multi-UAV](https://mavsdk.mavlink.io/)
- [Gazebo Multi-Robot](http://gazebosim.org/tutorials?tut=ros_multirobots)

### Exemples
- `example_px4_integrated.py` - Script principal multi-UAV
- `launch_multi_uav.sh` - Lanceur automatique
- `px4_config.py` - Configuration paramètres

### Support
- Forum PX4: https://discuss.px4.io/
- Issues GitHub: Créer un ticket avec logs
- Documentation en ligne: Cette README

## 🤝 Contribution

Pour améliorer cette intégration:
1. Tester avec différents nombres d'UAVs
2. Documenter les problèmes rencontrés
3. Proposer des optimisations
4. Partager vos scénarios de mission

## 📝 Notes Importantes

⚠️ **Limitations:**
- Maximum 10 UAVs recommandé (limite hardware)
- Performance dépend de votre CPU
- Gazebo peut ralentir avec >5 drones

✅ **Bonnes Pratiques:**
- Commencer avec 2-3 UAVs
- Surveiller les logs initialement
- Utiliser QGroundControl pour validation
- Sauvegarder configurations qui marchent

🎓 **Apprentissage:**
- Maîtriser 1 UAV d'abord
- Ajouter graduellement
- Comprendre les interactions
- Expérimenter avec stratégies

---

**Version:** 1.0  
**Date:** Novembre 2024  
**Compatible:** PX4 v1.14+, MAVSDK Python 2.0+
