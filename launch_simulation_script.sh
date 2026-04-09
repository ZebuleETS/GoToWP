#!/bin/bash

# Script de lancement Multi-UAV pour PX4 SITL + Gazebo
# Usage: ./launch_simulation_script.sh [nombre_uavs]

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "Lancement Simulation Multi-UAV PX4 SITL"
echo -e "==========================================${NC}"

# Nombre d'UAVs (par défaut 3, max 10)
NUM_UAVS=${1:-3}
if [ $NUM_UAVS -gt 10 ]; then
    echo -e "${RED}Maximum 10 UAVs supportés${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration: $NUM_UAVS UAVs${NC}"

# Répertoires
PX4_DIR="$HOME/GoToWP/PX4-Autopilot-soaring"
MAVSDK_DIR="$HOME/MAVSDK"
GOTOWP_DIR="$HOME/GoToWP"
XRCE_DDS_DIR="$HOME/Micro-XRCE-DDS-Agent"
AUTOSOARING_DIR="$GOTOWP_DIR/autosoaring"
AUTOSOARING_CONFIG_FILE="$AUTOSOARING_DIR/src/autosoaring_pkg/config/thermal_config.yaml"
LOG_DIR="$GOTOWP_DIR/multi_uav_logs"

# Configurer les chemins pour les plugins Gazebo custom
export GZ_SIM_SYSTEM_PLUGIN_PATH="${PX4_DIR}/Tools/simulation/gz/GZ_Plugins/liftdrag/build:${PX4_DIR}/Tools/simulation/gz/GZ_Plugins/liftdrag_advanced/build:${PX4_DIR}/Tools/simulation/gz/GZ_Plugins/MulticopterMotorModel/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_DESCRIPTOR_PATH="${PX4_DIR}/Tools/simulation/gz/GZ_Msgs/build${GZ_DESCRIPTOR_PATH:+:$GZ_DESCRIPTOR_PATH}"

# Vérifier que PX4-Autopilot (version mise à jour) existe
if [ ! -d "$PX4_DIR" ]; then
    echo -e "${RED}❌ PX4-Autopilot non trouvé dans $PX4_DIR${NC}"
    exit 1
fi
if [ ! -f "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
    echo -e "${RED}❌ PX4 SITL non compilé. Compilez d'abord :${NC}"
    echo "  cd $PX4_DIR && make px4_sitl_default"
    exit 1
fi
echo -e "${GREEN}✓ PX4-Autopilot (updated) trouvé : $PX4_DIR${NC}"
# Vérifier que MAVSDK existe
if [ ! -d "$MAVSDK_DIR" ]; then
    echo -e "${RED}❌ MAVSDK non trouvé dans $MAVSDK_DIR${NC}"
    echo "Clonez-le avec: git clone https://github.com/mavlink/MAVSDK.git"
    exit 1
fi

# Vérifier que Micro XRCE-DDS Agent existe
if [ ! -f "$XRCE_DDS_DIR/build/MicroXRCEAgent" ]; then
    echo -e "${RED}❌ Micro XRCE-DDS Agent non trouvé dans $XRCE_DDS_DIR/build/${NC}"
    echo "Compilez-le avec:"
    echo "  cd $XRCE_DDS_DIR && mkdir -p build && cd build"
    echo "  cmake .. && make -j\$(nproc)"
    exit 1
fi

# Nettoyer les processus précédents
echo -e "${YELLOW}Nettoyage des processus existants...${NC}"
pkill -9 -f mavsdk_server 2>/dev/null || true
pkill -9 -f MicroXRCEAgent 2>/dev/null || true
pkill -9 -f px4 2>/dev/null || true
pkill -9 -f gazebo 2>/dev/null || true
pkill -9 -f gzserver 2>/dev/null || true
pkill -9 -f gzclient 2>/dev/null || true
pkill -9 -f thermal_generator_node 2>/dev/null || true
pkill -9 -f thermal_detection_node 2>/dev/null || true
pkill -9 -f thermal_mapping_node 2>/dev/null || true
pkill -9 -f battery_manager_node 2>/dev/null || true

# Supprimer les logs
rm -rf $LOG_DIR
sleep 3

# Créer un répertoire pour les logs
mkdir -p $LOG_DIR
echo -e "${GREEN}Logs seront sauvegardés dans: $LOG_DIR${NC}"

# Tableau pour stocker les PIDs
declare -a PX4_PIDS
declare -a MAVSDK_PIDS
XRCE_DDS_PID=""
ROS2_LAUNCH_PID=""

# Fonction de nettoyage
cleanup () {
    echo -e "\n${YELLOW}Nettoyage en cours...${NC}"
    
    # Tuer tous les processus PX4
    for PID in "${PX4_PIDS[@]}"; do
        if [ ! -z "$PID" ]; then
            kill $PID 2>/dev/null || true
        fi
    done
    # Tuer tous les processus MAVSDK
    for PID in "${MAVSDK_PIDS[@]}"; do
        if [ ! -z "$PID" ]; then
            kill $PID 2>/dev/null || true
        fi
    done
    # Tuer le Micro XRCE-DDS Agent
    if [ ! -z "$XRCE_DDS_PID" ]; then
        kill $XRCE_DDS_PID 2>/dev/null || true
    fi
    # Tuer le launch ROS2 (thermal generator + autres nœuds)
    if [ ! -z "$ROS2_LAUNCH_PID" ]; then
        kill $ROS2_LAUNCH_PID 2>/dev/null || true
    fi
    
    # Tuer Gazebo, px4 orphelins et mavsdk_server restants
    pkill -9 -f mavsdk_server 2>/dev/null || true
    pkill -9 -f MicroXRCEAgent 2>/dev/null || true
    pkill -9 -f "bin/px4" 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f gazebo 2>/dev/null || true
    pkill -9 -f gzserver 2>/dev/null || true
    pkill -9 -f gzclient 2>/dev/null || true
    pkill -9 -f thermal_generator_node 2>/dev/null || true
    pkill -9 -f thermal_detection_node 2>/dev/null || true
    pkill -9 -f thermal_mapping_node 2>/dev/null || true
    pkill -9 -f battery_manager_node 2>/dev/null || true
    
    echo -e "${GREEN}✓ Nettoyage terminé${NC}"
    exit 0
}

# Capturer Ctrl+C
trap cleanup SIGINT SIGTERM

# ========== ÉTAPE 1 : Lancer TOUS les serveurs MAVSDK ==========
echo -e "\n${BLUE}=========================================="
echo "Lancement des serveurs MAVSDK"
echo -e "==========================================${NC}"

cd $MAVSDK_DIR

for ((i=0; i<$NUM_UAVS; i++)); do
    MAVLINK_UDP=$((14540 + i))
    MAVSDK_PORT=$((50051 + i))
    
    ./build/src/mavsdk_server/src/mavsdk_server udpin://0.0.0.0:$MAVLINK_UDP -p $MAVSDK_PORT \
        > $LOG_DIR/mavsdk_uav_${i}.log 2>&1 &
    
    MAVSDK_PIDS[$i]=$!
    echo -e "${GREEN}✓ MAVSDK server UAV $i : UDP $MAVLINK_UDP → port $MAVSDK_PORT (PID: ${MAVSDK_PIDS[$i]})${NC}"
done

sleep 2

# ========== ÉTAPE 2 : Lancer le Micro XRCE-DDS Agent ==========
echo -e "\n${BLUE}=========================================="
echo "Lancement du Micro XRCE-DDS Agent"
echo -e "==========================================${NC}"

XRCE_DDS_PORT=8888

$XRCE_DDS_DIR/build/MicroXRCEAgent udp4 -p $XRCE_DDS_PORT \
    > $LOG_DIR/xrce_dds_agent.log 2>&1 &

XRCE_DDS_PID=$!
echo -e "${GREEN}✓ Micro XRCE-DDS Agent : UDP port $XRCE_DDS_PORT (PID: $XRCE_DDS_PID)${NC}"

sleep 1

# ========== ÉTAPE 3 : Vérification Build PX4 ==========
echo -e "\n${BLUE}=========================================="
echo "Vérification PX4 SITL"
echo -e "==========================================${NC}"

cd $PX4_DIR

if [ -f "./build/px4_sitl_default/bin/px4" ]; then
    echo -e "${GREEN}✓ Build PX4 SITL existant trouvé${NC}"
else
    echo -e "${YELLOW}Build PX4 SITL en cours...${NC}"
    make px4_sitl_default > $LOG_DIR/px4_build.log 2>&1
    echo -e "${GREEN}✓ Build PX4 terminé${NC}"
fi

# ========== ÉTAPE 4 : Lancer toutes les instances PX4 ==========
echo -e "\n${BLUE}=========================================="
echo "Lancement de toutes les instances PX4"
echo -e "==========================================${NC}"

# Positions de spawn espacées pour éviter les collisions
SPAWN_POSITIONS=("0,0,0.5" "0,5,0.5" "0,-5,0.5" "5,0,0.5" "-5,0,0.5" "5,5,0.5" "-5,-5,0.5" "5,-5,0.5" "-5,5,0.5" "10,0,0.5")

for ((i=0; i<$NUM_UAVS; i++)); do
    INSTANCE=$i
    
    echo -e "\n${BLUE}--- UAV $i ---${NC}"
    echo "  Instance: $INSTANCE"
    echo "  Spawn: ${SPAWN_POSITIONS[$i]}"
    
    if [ $i -eq 0 ]; then
        # Instance 0 : lance Gazebo automatiquement (pas de STANDALONE)
        # PX4 détecte qu'aucun monde n'est lancé et démarre gz sim lui-même
        HEADLESS=1 \
        PX4_SYS_AUTOSTART=4003 \
        PX4_GZ_WORLD=default \
        PX4_GZ_MODEL_POSE="${SPAWN_POSITIONS[$i]}" \
        PX4_SIM_MODEL=gz_rc_cessna \
        ./build/px4_sitl_default/bin/px4 -i $INSTANCE \
            > $LOG_DIR/px4_uav_${i}.log 2>&1 &
        
        PX4_PIDS[$i]=$!
        echo -e "${GREEN}✓ UAV $i lancé + Gazebo (PID: ${PX4_PIDS[$i]})${NC}"
        
        # Attendre que Gazebo soit prêt avant de lancer les autres instances
        echo -e "${YELLOW}Attente de Gazebo + UAV 0 (max 120s)...${NC}"
        MAX_WAIT=120
        ELAPSED=0
        while [ $ELAPSED -lt $MAX_WAIT ]; do
            if gz topic -l 2>/dev/null | grep -q "clock"; then
                echo -e "${GREEN}✓ Gazebo prêt après ${ELAPSED}s${NC}"
                break
            fi
            if ! kill -0 ${PX4_PIDS[0]} 2>/dev/null; then
                echo -e "${RED}✗ PX4 UAV 0 a échoué. Vérifiez $LOG_DIR/px4_uav_0.log${NC}"
                cleanup
                exit 1
            fi
            sleep 3
            ELAPSED=$((ELAPSED + 3))
            echo -e "${YELLOW}  Attente Gazebo... (${ELAPSED}s)${NC}"
        done
        
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo -e "${RED}✗ Timeout : Gazebo n'a pas démarré en ${MAX_WAIT}s${NC}"
            cleanup
            exit 1
        fi
        
        sleep 5
    else
        # Instances 1+ : mode standalone, se connecte au Gazebo existant
        PX4_GZ_STANDALONE=1 \
        PX4_SYS_AUTOSTART=4003 \
        PX4_GZ_MODEL_POSE="${SPAWN_POSITIONS[$i]}" \
        PX4_SIM_MODEL=gz_rc_cessna \
        ./build/px4_sitl_default/bin/px4 -i $INSTANCE \
            > $LOG_DIR/px4_uav_${i}.log 2>&1 &
        
        PX4_PIDS[$i]=$!
        echo -e "${GREEN}✓ UAV $i lancé en standalone (PID: ${PX4_PIDS[$i]})${NC}"
        
        # Attendre entre chaque lancement
        sleep 5
    fi
done

# Attendre que tous soient initialisés (EKF heading stabilisation)
echo -e "\n${YELLOW}Attente de l'initialisation complète (5s)...${NC}"
sleep 5

# ========== ÉTAPE 5 : Lancer les nœuds ROS2 AutoSoaring ==========
echo -e "\n${BLUE}=========================================="
echo "Lancement du thermal generator ROS2"
echo -e "==========================================${NC}"

# Source ROS2 Jazzy + workspace autosoaring
source /opt/ros/jazzy/setup.bash
if [ -f "$AUTOSOARING_DIR/install/setup.bash" ]; then
    source $AUTOSOARING_DIR/install/setup.bash
    echo -e "${GREEN}✓ Workspace autosoaring sourcé${NC}"
else
    echo -e "${RED}❌ Workspace autosoaring non compilé. Compilez d'abord :${NC}"
    echo "  cd $AUTOSOARING_DIR && colcon build"
    cleanup
    exit 1
fi

if [ ! -f "$AUTOSOARING_CONFIG_FILE" ]; then
    echo -e "${RED}❌ Fichier de config thermal introuvable : $AUTOSOARING_CONFIG_FILE${NC}"
    cleanup
    exit 1
fi

# Lancer le thermal generator node (crée les thermiques dans Gazebo + publie sur ROS2)
ros2 launch autosoaring_pkg autosoaring_launch.py mode:=generator \
    config_file:=$AUTOSOARING_CONFIG_FILE \
    > $LOG_DIR/ros2_autosoaring.log 2>&1 &
ROS2_LAUNCH_PID=$!
echo -e "${GREEN}✓ ROS2 thermal_generator_node lancé (PID: $ROS2_LAUNCH_PID)${NC}"

# Attendre que le topic /thermal_snapshot soit disponible
echo -e "${YELLOW}Attente du topic /thermal_snapshot (max 30s)...${NC}"
MAX_WAIT=30
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ros2 topic list 2>/dev/null | grep -q "/thermal_snapshot"; then
        echo -e "${GREEN}✓ Topic /thermal_snapshot disponible après ${ELAPSED}s${NC}"
        break
    fi
    if ! kill -0 $ROS2_LAUNCH_PID 2>/dev/null; then
        echo -e "${RED}✗ ROS2 launch a échoué. Vérifiez $LOG_DIR/ros2_autosoaring.log${NC}"
        cleanup
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo -e "${YELLOW}⚠  Topic /thermal_snapshot non détecté en ${MAX_WAIT}s — le nœud tourne peut-être quand même${NC}"
fi

sleep 2

# ========== ÉTAPE 6 : Vérification ==========
echo -e "\n${BLUE}Vérification des processus...${NC}"
ALL_OK=true

# Vérifier Micro XRCE-DDS Agent
if ps -p $XRCE_DDS_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Micro XRCE-DDS Agent (PID $XRCE_DDS_PID) : OK${NC}"
else
    echo -e "${RED}✗ Micro XRCE-DDS Agent (PID $XRCE_DDS_PID) : Arrêté (vérifier $LOG_DIR/xrce_dds_agent.log)${NC}"
    ALL_OK=false
fi

# Vérifier MAVSDK
for i in "${!MAVSDK_PIDS[@]}"; do
    PID=${MAVSDK_PIDS[$i]}
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ MAVSDK UAV $i (PID $PID) : OK${NC}"
    else
        echo -e "${RED}✗ MAVSDK UAV $i (PID $PID) : Arrêté${NC}"
        ALL_OK=false
    fi
done

# Vérifier PX4
for i in "${!PX4_PIDS[@]}"; do
    PID=${PX4_PIDS[$i]}
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PX4 UAV $i (PID $PID) : OK${NC}"
    else
        echo -e "${RED}✗ PX4 UAV $i (PID $PID) : Arrêté (vérifier $LOG_DIR/px4_uav_${i}.log)${NC}"
        ALL_OK=false
    fi
done

# Vérifier ROS2 thermal generator
if ps -p $ROS2_LAUNCH_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ROS2 AutoSoaring (PID $ROS2_LAUNCH_PID) : OK${NC}"
else
    echo -e "${RED}✗ ROS2 AutoSoaring (PID $ROS2_LAUNCH_PID) : Arrêté (vérifier $LOG_DIR/ros2_autosoaring.log)${NC}"
    ALL_OK=false
fi

# Vérifier les messages Gazebo custom (thermal)
echo -e "\n${BLUE}Vérification des messages Gazebo thermiques...${NC}"
if gz msg -l 2>/dev/null | grep -q "gz.msgs.Thermal"; then
    echo -e "${GREEN}✓ Types gz.msgs.Thermal / ThermalGroup enregistrés${NC}"
else
    echo -e "${RED}✗ Types gz.msgs.Thermal non trouvés — vérifiez GZ_DESCRIPTOR_PATH${NC}"
    echo -e "${YELLOW}  Actuel: GZ_DESCRIPTOR_PATH=$GZ_DESCRIPTOR_PATH${NC}"
    ALL_OK=false
fi

if gz topic -l 2>/dev/null | grep -q "thermal_updrafts"; then
    echo -e "${GREEN}✓ Topic /world/default/thermal_updrafts disponible${NC}"
    # Tenter de capturer un message (timeout 5s)
    THERMAL_DATA=$(timeout 5 gz topic -e -n 1 -t /world/default/thermal_updrafts 2>/dev/null || true)
    if [ -n "$THERMAL_DATA" ]; then
        echo -e "${GREEN}✓ Messages thermiques reçus sur Gazebo transport${NC}"
    else
        echo -e "${YELLOW}⚠  Topic existe mais aucun message reçu en 5s (le générateur publie peut-être pas encore)${NC}"
    fi
else
    echo -e "${YELLOW}⚠  Topic /world/default/thermal_updrafts pas encore disponible (le générateur démarrera bientôt)${NC}"
fi

if [ "$ALL_OK" = false ]; then
    echo -e "${RED}Certains processus ne sont pas lancés correctement${NC}"
    echo "Vérifiez les logs dans $LOG_DIR"
    cleanup
    exit 1
fi

echo -e "\n${GREEN}=========================================="
echo "✓ Tous les UAVs sont prêts!"
echo -e "==========================================${NC}"

# Afficher les informations de connexion
echo -e "\n${BLUE}Informations de connexion:${NC}"
for ((i=0; i<$NUM_UAVS; i++)); do
    UDP_PORT=$((14540 + i))
    SDK_PORT=$((50051 + i))
    echo "  UAV $i: MAVLink udp://:$UDP_PORT → MAVSDK port $SDK_PORT"
done

echo -e "\n${YELLOW}Pour QGroundControl, connectez sur udp://:14550${NC}"
echo -e "${BLUE}Topics ROS2 thermiques :${NC}"
echo "  /thermal_snapshot   — Toutes les thermiques actives (pour l'algorithme)"
echo "  /generated_thermals — Nouvelles thermiques (pour le mapping)"
echo "  /thermal_removed    — IDs des thermiques expirées"

# ========== ÉTAPE 7 : Lancer la simulation Python ==========
echo -e "\n${BLUE}=========================================="
echo "Lancement de la simulation Python"
echo -e "==========================================${NC}"

sleep 3

echo -e "${GREEN}Prêt pour le décollage!${NC}"
echo -e "${YELLOW}Appuyez sur Entrée pour lancer la simulation Python...${NC}"
read

# Lancer le script Python
# On source ROS2 + autosoaring pour que rclpy et thermal_ros_bridge soient accessibles
source /opt/ros/jazzy/setup.bash
source $AUTOSOARING_DIR/install/setup.bash
source $GOTOWP_DIR/Mav/bin/activate
python3 $GOTOWP_DIR/dronePx4.py

# Nettoyage final
cleanup
