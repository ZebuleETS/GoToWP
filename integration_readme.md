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
python3 example_px4_integrated.py
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
