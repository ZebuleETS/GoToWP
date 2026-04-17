**Auteurs**:

Nicolas Plourde (nicolas.plourde.1@ens.etsmtl.ca), École de technologie supérieure
Houssem-Eddine Mohamadi (houssem-eddine.mohamadi@etsmtl.net), École de technologie supérieure
Nadjia Kara (https://www.etsmtl.ca/en/labs/imagin-lab), École de technologie supérieure

**Contributeurs**:

Nicolas Plourde (nicolas.plourde.1@ens.etsmtl.ca), École de technologie supérieure

# GoToWP — Simulation Multi-UAV Planeurs Autonomes avec Thermiques

Projet de simulation et d'intégration d'un essaim de planeurs autonomes (fixed-wing) capables de détecter et exploiter des thermiques dans un environnement **PX4 SITL + Gazebo Harmonic + ROS 2 Jazzy**.

Le projet combine :
- Un noyau Python de planification de trajectoire, de gestion énergétique et de détection thermique ([GoToWP.py](GoToWP.py), [compute.py](compute.py), [trajectory.py](trajectory.py), [thermal.py](thermal.py)).
- Un pont vers PX4 SITL via MAVSDK pour piloter des drones réels/simulés ([dronePx4.py](dronePx4.py)).
- Un package ROS 2 `autosoaring_pkg` qui génère, détecte, cartographie les thermiques et gère la batterie ([autosoaring/](autosoaring/)).
- Un fork de PX4-Autopilot avec plugins Gazebo custom (lift/drag avancé, messages `gz.msgs.Thermal`) ([PX4-Autopilot-soaring/](PX4-Autopilot-soaring/)).

---

## Table des matières

1. [Architecture](#architecture)
2. [Prérequis système](#prérequis-système)
3. [Installation complète](#installation-complète)
4. [Configuration](#configuration)
5. [Lancement de la simulation](#lancement-de-la-simulation)
6. [Structure du projet](#structure-du-projet)
7. [Utilisation des modules](#utilisation-des-modules)
8. [Débogage](#débogage)
9. [Ressources](#ressources)

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Simulation Python                       │
│  dronePx4.py (orchestrateur)                             │
│   ├─ GoToWP.py      — planification waypoints            │
│   ├─ trajectory.py  — évaluation trajectoires            │
│   ├─ thermal.py     — carte & détection thermiques       │
│   ├─ compute.py     — géométrie, aéro, énergétique       │
│   └─ Scenario.py    — scénarios de test & métriques      │
└─────────────────┬────────────────────────────────────────┘
                  │ MAVSDK (asyncio)              ROS 2
                  ▼                                  ▼
        ┌───────────────────┐          ┌──────────────────────┐
        │  mavsdk_server    │          │ thermal_ros_bridge   │
        │  (1 par UAV)      │          │ rclpy ↔ ThermalMap   │
        └─────────┬─────────┘          └──────────┬───────────┘
                  │ MAVLink UDP                    │ Topics ROS 2
                  ▼                                ▼
      ┌────────────────────────────────────────────────────┐
      │        PX4 SITL (N instances) + ROS 2 Jazzy        │
      │   autosoaring_pkg : generator / detection /        │
      │                     mapping / battery              │
      └───────────────────────┬────────────────────────────┘
                              │ Gazebo Transport
                              ▼
               ┌──────────────────────────────┐
               │  Gazebo Harmonic (gz sim)    │
               │   + plugins liftdrag custom  │
               │   + messages Thermal custom  │
               └──────────────────────────────┘
```

### Systèmes de coordonnées

| Repère | Description |
|---|---|
| **XYZ** (simulation) | X=Est, Y=Nord, Z=Altitude AGL (m) — origine arbitraire |
| **NED** (PX4) | Nord, Est, Bas — origine position home |
| **GPS** | lat/lon WGS84 + altitude MSL |

La conversion est gérée par [dronePx4.py](dronePx4.py) et utilise [pymap3d](https://github.com/geospace-code/pymap3d).

---

## Prérequis système

| Composant | Version recommandée |
|---|---|
| OS | Ubuntu 22.04 / 24.04 (ou WSL2) |
| Python | 3.12 |
| ROS 2 | Jazzy |
| Gazebo | Harmonic (gz sim) |
| RAM | ≥ 8 Go (16 Go pour ≥ 5 UAVs) |
| Disque | ≥ 30 Go |

### Dépendances externes à installer séparément

- **PX4-Autopilot-soaring** — fork PX4 inclus en sous-module (voir [Installation](#2-compilation-du-fork-px4).
- **MAVSDK** (serveur C++) — à cloner à `~/MAVSDK`.
- **Micro-XRCE-DDS Agent** — à cloner à `~/Micro-XRCE-DDS-Agent`.

---

## Installation complète

### 1. Cloner le projet avec ses sous-modules

```bash
cd ~
git clone --recursive <url_du_repo> GoToWP
cd GoToWP
```

Si le clone a déjà été fait sans `--recursive` :

```bash
git submodule update --init --recursive
```

Les deux sous-modules sont définis dans [.gitmodules](.gitmodules) :

| Sous-module | Chemin | Rôle |
|---|---|---|
| `autosoaring` | [autosoaring/](autosoaring/) | Package ROS 2 (détection, mapping, batterie, générateur de thermiques) |
| `PX4-Autopilot-soaring` | [PX4-Autopilot-soaring/](PX4-Autopilot-soaring/) | Fork PX4 avec plugins Gazebo custom (liftdrag avancé, messages thermiques) |

### 2. Compilation du fork PX4

```bash
cd ~/GoToWP/PX4-Autopilot-soaring

# Installer les dépendances système (une seule fois)
bash ./Tools/setup/ubuntu.sh

# Compiler SITL (long : 15-30 min la 1ère fois)
make px4_sitl_default
```

Compilation des plugins Gazebo custom (liftdrag, messages thermiques) :

```bash
# Messages thermiques custom
cd ~/GoToWP/PX4-Autopilot-soaring/Tools/simulation/gz/GZ_Msgs
mkdir -p build && cd build && cmake .. && make

# Plugin liftdrag avancé
cd ~/GoToWP/PX4-Autopilot-soaring/Tools/simulation/gz/GZ_Plugins/liftdrag_advanced
mkdir -p build && cd build && cmake .. && make
```

### 3. Installation de MAVSDK (serveur C++)

```bash
cd ~
git clone https://github.com/mavlink/MAVSDK.git --recursive
cd MAVSDK
cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Release -DBUILD_MAVSDK_SERVER=ON
cmake --build build -j$(nproc)
```

Le binaire attendu est `~/MAVSDK/build/src/mavsdk_server/src/mavsdk_server`.

### 4. Installation de Micro-XRCE-DDS Agent

```bash
cd ~
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig /usr/local/lib/
```

### 5. Environnement Python (venv `Mav`)

Le projet utilise un venv local nommé `Mav/` à la racine de [GoToWP/](.).

```bash
cd ~/GoToWP
python3 -m venv Mav
source Mav/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Contenu de [requirements.txt](requirements.txt) :

```
mavsdk>=3.5.0
numpy>=2.3.4
scipy>=1.16.3
pymap3d>=3.2.0
protobuf>=6.33.0
```

### 6. Installation de ROS 2 Jazzy et du package `autosoaring_pkg`

Installer ROS 2 Jazzy selon la procédure officielle : <https://docs.ros.org/en/jazzy/Installation.html>

Puis compiler le workspace autosoaring :

```bash
cd ~/GoToWP/autosoaring
source /opt/ros/jazzy/setup.bash
colcon build --packages-select autosoaring_pkg
source install/setup.bash
```

> Voir aussi [autosoaring/README.md](autosoaring/README.md) pour plus de détails.

---

## Configuration

### Paramètres PX4 (Fixed Wing)

Paramètres ajoutés au model du planeur:

| Paramètre | Valeur | Description |
|---|---|---|
| `FW_AIRSPD_MIN` | 8.0 m/s | Vitesse de décrochage |
| `FW_AIRSPD_MAX` | 30.0 m/s | Vitesse maximale |
| `FW_AIRSPD_TRIM` | 18.0 m/s | Vitesse de croisière |
| `NAV_LOITER_RAD` | 80.0 m | Rayon de loiter |
| `NAV_ACC_RAD` | 20.0 m | Rayon d'acceptation waypoint |

### Générateur de thermiques

Fichier : [autosoaring/src/autosoaring_pkg/config/thermal_config.yaml](autosoaring/src/autosoaring_pkg/config/thermal_config.yaml)

Il contrôle le nombre, la distribution spatiale, la force et la durée de vie des thermiques publiées sur le topic Gazebo `/world/default/thermal_updrafts` et sur le topic ROS 2 `/thermal_snapshot`.

### Paramètres aérodynamiques du planeur

Dans [dronePx4.py](dronePx4.py) (dictionnaire `UAV_data`) :

```python
UAV_data['maximum_battery_capacity'] = 10.0  # Ah
UAV_data['empty_weight']             = 1.6   # kg
UAV_data['wing_area']                = 0.5   # m²
UAV_data['wing_aspect_ratio']        = 15.7
UAV_data['max_airspeed']             = 30.0  # m/s
UAV_data['min_airspeed']             = 8.0   # m/s
```

---

## Lancement de la simulation

### Lancement automatique (recommandé)

```bash
cd ~/GoToWP
chmod +x launch_simulation_script.sh
./launch_simulation_script.sh 3      # 3 UAVs (défaut)
./launch_simulation_script.sh 5      # 5 UAVs
```

Le script [launch_simulation_script.sh](launch_simulation_script.sh) exécute automatiquement :

1. Nettoyage des processus précédents (px4, gazebo, mavsdk_server, XRCE-DDS).
2. Démarrage de N instances `mavsdk_server` (UDP 14540+i → gRPC 50051+i).
3. Démarrage du `MicroXRCEAgent` (UDP 8888) pour uXRCE-DDS.
4. Vérification / compilation de PX4 SITL.
5. Lancement de l'UAV 0 avec Gazebo (modèle `gz_rc_cessna`), puis des UAVs suivants en mode `PX4_GZ_STANDALONE=1`.
6. Lancement du nœud ROS 2 `thermal_generator` (`ros2 launch autosoaring_pkg autosoaring_launch.py mode:=generator`).
7. Vérification des topics Gazebo (`/world/default/thermal_updrafts`) et ROS 2 (`/thermal_snapshot`).
8. Lancement de [dronePx4.py](dronePx4.py) dans le venv `Mav/` une fois que l'utilisateur appuie sur Entrée.

### Lancement manuel (pour debug)

Terminal 1 — Gazebo + PX4 UAV 0 :
```bash
cd ~/GoToWP/PX4-Autopilot-soaring
HEADLESS=1 PX4_SYS_AUTOSTART=4003 PX4_GZ_WORLD=default \
PX4_GZ_MODEL_POSE="0,0,0.5" PX4_SIM_MODEL=gz_rc_cessna \
./build/px4_sitl_default/bin/px4 -i 0
```

Terminal 2+ — PX4 UAV 1..N (standalone) :
```bash
PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4003 \
PX4_GZ_MODEL_POSE="0,5,0.5" PX4_SIM_MODEL=gz_rc_cessna \
./build/px4_sitl_default/bin/px4 -i 1
```

Terminal 3 — MAVSDK server (un par UAV) :
```bash
~/MAVSDK/build/src/mavsdk_server/src/mavsdk_server udpin://0.0.0.0:14540 -p 50051
```

Terminal 4 — Micro XRCE-DDS Agent :
```bash
~/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent udp4 -p 8888
```

Terminal 5 — Nœuds ROS 2 autosoaring :
```bash
source /opt/ros/jazzy/setup.bash
source ~/GoToWP/autosoaring/install/setup.bash
ros2 launch autosoaring_pkg autosoaring_launch.py \
    mode:=generator \
    config_file:=$HOME/GoToWP/autosoaring/src/autosoaring_pkg/config/thermal_config.yaml
```

Terminal 6 — Simulation Python :
```bash
source /opt/ros/jazzy/setup.bash
source ~/GoToWP/autosoaring/install/setup.bash
source ~/GoToWP/Mav/bin/activate
python3 ~/GoToWP/dronePx4.py
```

### Ports utilisés

| UAV | MAVLink UDP | MAVSDK gRPC | PX4 instance |
|---|---|---|---|
| 0 | 14540 | 50051 | -i 0 |
| 1 | 14541 | 50052 | -i 1 |
| N | 14540+N | 50051+N | -i N |

QGroundControl : `udp://:14550` (tous les UAVs multiplexés).

---

## Structure du projet

```
GoToWP/
├── README.md                      # Ce fichier
├── integration_readme.md          # Guide détaillé d'intégration PX4
├── requirements.txt               # Dépendances Python (venv Mav)
├── launch_simulation_script.sh    # Orchestrateur complet
├── .gitmodules
│
├── Mav/                           # venv Python (non versionné)
│
├── GoToWP.py                      # Planification waypoints multi-UAV
├── compute.py                     # Géométrie, aéro, énergie, A*
├── trajectory.py                  # Évaluation et génération de trajectoires
├── thermal.py                     # Carte de thermiques, détection
├── thermal_ros_bridge.py          # Pont ROS 2 ↔ ThermalMap
├── Scenario.py                    # Scénarios de test, métriques
├── dronePx4.py                    # Orchestrateur principal (asyncio + MAVSDK)
├── example.py                     # Exemple simulation pure (sans PX4)
│
├── log/                           # Logs FLT_track
├── multi_uav_logs/                # Logs multi-UAV (générés par le script)
│
├── autosoaring/                   # Sous-module : package ROS 2
│   ├── README.md
│   ├── requirements.txt
│   ├── setup_autosoaring.sh
│   ├── start_autosoaring.sh
│   └── src/autosoaring_pkg/
│       ├── autosoaring_pkg/
│       │   ├── thermal_generator_node.py   # Génère thermiques Gazebo
│       │   ├── thermal_detection_node.py   # Détecte thermiques en vol
│       │   ├── thermal_mapping_node.py     # Cartographie et visualisation
│       │   └── battery_manager_node.py     # Gestion batterie
│       ├── launch/autosoaring_launch.py
│       ├── config/thermal_config.yaml
│       └── GZ_Msgs/                         # Messages Gazebo custom
│
└── PX4-Autopilot-soaring/         # Sous-module : fork PX4
    └── Tools/simulation/gz/
        ├── GZ_Msgs/               # Messages thermal.proto
        └── GZ_Plugins/
            ├── liftdrag/
            ├── liftdrag_advanced/ # Plugin aérodynamique custom
            └── MulticopterMotorModel/
```

---

## Utilisation des modules

### Module `GoToWP` — planification de waypoints

```python
from GoToWP import gotoWaypointMulti

waypoints = gotoWaypointMulti(
    uav_positions, targets, obstacles, params
)
```

### Module `trajectory` — évaluation de trajectoires

```python
from trajectory import (
    TrajectoryEvaluator,
    generate_all_trajectories,
    LawnMowerTrajectory,
)

trajectories = generate_all_trajectories(start, end, obstacles)
evaluator = TrajectoryEvaluator(UAV_data, params)
best = evaluator.evaluate(trajectories)
```

### Module `thermal` — carte et détection

```python
from thermal import ThermalMap, detect_thermal_at_position

tmap = ThermalMap()
tmap.add_thermal(x=1000, y=2000, radius=150, strength=2.5)

detection = detect_thermal_at_position(tmap, drone_x, drone_y, drone_z)
```

### Module `thermal_ros_bridge` — pont ROS 2

```python
import rclpy
from thermal_ros_bridge import ThermalROSBridge

rclpy.init()
bridge = ThermalROSBridge(thermal_map)
rclpy.spin(bridge)
```

S'abonne aux topics :
- `/thermal_snapshot` — toutes les thermiques actives
- `/generated_thermals` — nouvelles thermiques
- `/thermal_removed` — IDs expirés

### Nœuds ROS 2 du package `autosoaring_pkg`

```bash
# Lancement groupé
ros2 launch autosoaring_pkg autosoaring_launch.py

# Ou nœuds individuels
ros2 run autosoaring_pkg thermal_generator_node /chemin/config.yaml
ros2 run autosoaring_pkg thermal_detection_node
ros2 run autosoaring_pkg thermal_mapping_node
ros2 run autosoaring_pkg battery_manager_node
```

### Exemple minimal (sans PX4)

```bash
source Mav/bin/activate
python3 example.py
```

[example.py](example.py) exécute une simulation pure Python (sans PX4/Gazebo) pour tester la planification, les thermiques et les trajectoires.

---

## Débogage

### Vérifications rapides

```bash
# MAVLink ports actifs
netstat -ln | grep 1454

# Topics Gazebo
gz topic -l | grep thermal

# Types de messages Gazebo custom
gz msg -l | grep Thermal

# Topics ROS 2
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep thermal
ros2 topic echo /thermal_snapshot
```

### Problèmes fréquents

**PX4 ne démarre pas**
```bash
cd ~/GoToWP/PX4-Autopilot-soaring
make clean && make px4_sitl_default
```

**MAVSDK timeout à la connexion**
- Vérifier `~/GoToWP/multi_uav_logs/mavsdk_uav_*.log`
- Attendre 10-15 s après PX4
- Vérifier que le port UDP 14540+i est libre avant le lancement

**Plugin liftdrag non chargé**
- Vérifier que `GZ_SIM_SYSTEM_PLUGIN_PATH` pointe bien vers les `build/` des plugins compilés
- Recompiler le plugin :
  ```bash
  cd ~/GoToWP/PX4-Autopilot-soaring/Tools/simulation/gz/GZ_Plugins/liftdrag_advanced/build
  cmake .. && make
  ```

**Erreur protobuf `File already exists in database: gz/msgs/thermal.proto`**
- Forcer un chemin unique via `GZ_DESCRIPTOR_PATH` (déjà fait dans [launch_simulation_script.sh](launch_simulation_script.sh)).

**Drone ne décolle pas**
- Vérifier le GPS fix dans QGroundControl
- Vérifier les paramètres `FW_*` (voir [Configuration](#configuration))
- Regarder les logs PX4 : `multi_uav_logs/px4_uav_*.log`

### Logs

```
multi_uav_logs/
├── gazebo.log
├── xrce_dds_agent.log
├── mavsdk_uav_0.log, mavsdk_uav_1.log, ...
├── px4_uav_0.log, px4_uav_1.log, ...
└── ros2_autosoaring.log
```

Logs Python de la simulation : dossier [log/](log/) (FLT_track par UAV).

---

## Ressources

- [autosoaring/README.md](autosoaring/README.md) — documentation du package ROS 2
- [Documentation PX4](https://docs.px4.io/)
- [MAVSDK Python](https://mavsdk.mavlink.io/main/en/python/)
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/)
- [Gazebo Harmonic](https://gazebosim.org/docs/harmonic)
- [QGroundControl](http://qgroundcontrol.com/)
- [Forum PX4](https://discuss.px4.io/)
