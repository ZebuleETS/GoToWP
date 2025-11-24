#!/bin/bash

# Script de lancement Multi-UAV pour PX4 SITL + Gazebo
# Usage: ./launch_multi_uav.sh [nombre_uavs]

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

# Vérifier que PX4-Autopilot existe
PX4_DIR="$HOME/PX4-Autopilot"
if [ ! -d "$PX4_DIR" ]; then
    echo -e "${RED}❌ PX4-Autopilot non trouvé dans $PX4_DIR${NC}"
    echo "Clonez-le avec: git clone https://github.com/PX4/PX4-Autopilot.git --recursive"
    exit 1
fi
# Vérifier que MAVSDK existe
MAVSDK_DIR="$HOME/MAVSDK"
if [ ! -d "$MAVSDK_DIR" ]; then
    echo -e "${RED}❌ MAVSDK non trouvé dans $MAVSDK_DIR${NC}"
    echo "Clonez-le avec: git clone https://github.com/mavlink/MAVSDK.git"
    exit 1
fi

# Nettoyer les processus précédents
echo -e "${YELLOW}Nettoyage des processus existants...${NC}"
pkill -9 -f px4 2>/dev/null || true
pkill -9 -f gazebo 2>/dev/null || true
pkill -9 -f gzserver 2>/dev/null || true
pkill -9 -f gzclient 2>/dev/null || true
sleep 3

# Créer un répertoire pour les logs
LOG_DIR="./multi_uav_logs"
mkdir -p $LOG_DIR
echo -e "${GREEN}Logs seront sauvegardés dans: $LOG_DIR${NC}"

# Tableau pour stocker les PIDs
declare -a PX4_PIDS

# Fonction pour lancer un UAV
launch_uav() {
    local UAV_ID=$1
    local INSTANCE=$UAV_ID
    
    # Ports pour chaque UAV
    local MAVLINK_UDP_PORT=$((14540 + UAV_ID))
    local MAVLINK_TCP_PORT=$((4560 + UAV_ID))
    local SIM_PORT=$((14560 + UAV_ID))
    
    # Position de spawn (espacer les drones)
    local SPAWN_X=$((UAV_ID % 3))
    local SPAWN_Y=$((UAV_ID / 3))
    
    echo -e "${BLUE}Lancement UAV $UAV_ID...${NC}"
    echo "  Instance: $INSTANCE"
    echo "  MAVLink UDP: $MAVLINK_UDP_PORT"
    echo "  Position spawn: ($SPAWN_X, $SPAWN_Y, 0)"
    
    cd $PX4_DIR
    
    # Variables d'environnement pour cette instance
    export PX4_SIM_MODEL=plane
    export PX4_INSTANCE=$INSTANCE
    
    # Commande de lancement
    if [ $UAV_ID -eq 0 ]; then
        # Premier UAV lance Gazebo
        DONT_RUN=1 make px4_sitl gazebo_plane___$INSTANCE \
            > $LOG_DIR/uav_${UAV_ID}.log 2>&1 &
    else
        # UAVs suivants utilisent le même Gazebo
        DONT_RUN=1 make px4_sitl gazebo_plane___$INSTANCE \
            > $LOG_DIR/uav_${UAV_ID}.log 2>&1 &
    fi
    
    local PID=$!
    PX4_PIDS[$UAV_ID]=$PID
    
    echo -e "${GREEN}  ✓ UAV $UAV_ID lancé (PID: $PID)${NC}"
    
    # Attendre un peu avant de lancer le suivant
    sleep 5
}

# Fonction de nettoyage
cleanup() {
    echo -e "\n${YELLOW}Nettoyage en cours...${NC}"
    
    # Tuer tous les processus PX4
    for PID in "${PX4_PIDS[@]}"; do
        if [ ! -z "$PID" ]; then
            kill $PID 2>/dev/null || true
        fi
    done
    
    # Tuer Gazebo
    pkill -9 -f gazebo 2>/dev/null || true
    pkill -9 -f gzserver 2>/dev/null || true
    pkill -9 -f gzclient 2>/dev/null || true
    
    echo -e "${GREEN}✓ Nettoyage terminé${NC}"
    exit 0
}

# Capturer Ctrl+C
trap cleanup SIGINT SIGTERM

# Lancer le premier UAV avec Gazebo
echo -e "\n${BLUE}=========================================="
echo "Lancement du premier UAV avec Gazebo"
echo -e "==========================================${NC}"

cd $PX4_DIR

# Lancer Gazebo avec le premier drone
xterm -hold -e make px4_sitl gz_rc_cessna && /bin/bash >> $LOG_DIR/gazebo.log 2>&1 &
GAZEBO_PID=$!
echo -e "${GREEN}✓ Gazebo lancé (PID: $GAZEBO_PID)${NC}"
sleep 10

# Lancer les instances PX4 pour chaque UAV
echo -e "\n${BLUE}=========================================="
echo "Lancement des instances PX4"
echo -e "==========================================${NC}"

for ((i=1; i<$NUM_UAVS; i++)); do
    # Calculer les ports
    MAVLINK_UDP=$((14540 + i))
    MAVSDK_PORT=$((50051 + i))
    INSTANCE=$i
    
    # Position de spawn
    SPAWN_X=$(echo "scale=1; ($i % 3) * 2" | bc)
    SPAWN_Y=$(echo "scale=1; ($i / 3) * 2" | bc)
    SPAWN_Z=400
    
    echo -e "\n${BLUE}--- UAV $i ---${NC}"
    echo "Instance: $INSTANCE"
    echo "MAVLink UDP: $MAVLINK_UDP"
    echo "MAVSDK Port: $MAVSDK_PORT"
    echo "Spawn position: ($SPAWN_X, $SPAWN_Y)"

    cd $MAVSDK_DIR

    # Lancer MAVSDK server pour cette instance
    xterm -hold -e "./build/src/mavsdk_server/src/mavsdk_server udpin://0.0.0.0:$MAVLINK_UDP -p $MAVSDK_PORT"
    
    cd $PX4_DIR
    
    # Lancer PX4 pour cette instance
    xterm -hold -e PX4_GZ_STANDALONE=1 \
    PX4_SYS_AUTOSTART=4003 \
    PX4_GZ_MODEL_POSE="$SPAWN_X, $SPAWN_Y" \
    PX4_SIM_MODEL=gz_rc_cessna \
    ./build/px4_sitl_default/bin/px4 \
        -i $INSTANCE && /bin/bash \
        >> $LOG_DIR/px4_uav_${i}.log 2>&1 &
    
    PID=$!
    PX4_PIDS[$i]=$PID
    
    echo -e "${GREEN}✓ UAV $i lancé (PID: $PID)${NC}"
    
    # Attendre entre chaque lancement
    sleep 5
done

# Attendre que tous soient initialisés
echo -e "\n${YELLOW}Attente de l'initialisation complète (15s)...${NC}"
sleep 15

# Vérifier que tous les processus tournent
echo -e "\n${BLUE}Vérification des processus...${NC}"
ALL_OK=true
for i in "${!PX4_PIDS[@]}"; do
    PID=${PX4_PIDS[$i]}
    if ps -p $PID > /dev/null; then
        echo -e "${GREEN}✓ UAV $i (PID $PID) : En cours d'exécution${NC}"
    else
        echo -e "${RED}✗ UAV $i (PID $PID) : Arrêté${NC}"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo -e "${RED}Certains UAVs ne sont pas lancés correctement${NC}"
    echo "Vérifiez les logs dans $LOG_DIR"
    cleanup
    exit 1
fi

echo -e "\n${GREEN}=========================================="
echo "✓ Tous les UAVs sont prêts!"
echo -e "==========================================${NC}"

# Afficher les informations de connexion
echo -e "\n${BLUE}Informations de connexion MAVLink:${NC}"
for ((i=0; i<$NUM_UAVS; i++)); do
    PORT=$((14540 + i))
    echo "  UAV $i: udp://:$PORT"
done

echo -e "\n${YELLOW}Pour QGroundControl, connectez sur udp://:14550${NC}"

# Lancer la simulation Python
echo -e "\n${BLUE}=========================================="
echo "Lancement de la simulation Python"
echo -e "==========================================${NC}"

cd - > /dev/null

# Attendre un peu avant de lancer Python
sleep 5

echo -e "${GREEN}Prêt pour le décollage!${NC}"
echo -e "${YELLOW}Appuyez sur Entrée pour lancer la simulation Python...${NC}"
read

# Lancer le script Python
python3 /home/pix4/GoToWP/dronePx4.py

# Nettoyage final
cleanup
