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
PX4_DIR="$HOME/PX4-Autopilot"
MAVSDK_DIR="$HOME/MAVSDK"
GOTOWP_DIR="$HOME/GoToWP"
LOG_DIR="$GOTOWP_DIR/multi_uav_logs"

# Vérifier que PX4-Autopilot existe
if [ ! -d "$PX4_DIR" ]; then
    echo -e "${RED}❌ PX4-Autopilot non trouvé dans $PX4_DIR${NC}"
    echo "Clonez-le avec: git clone https://github.com/PX4/PX4-Autopilot.git --recursive"
    exit 1
fi
# Vérifier que MAVSDK existe
if [ ! -d "$MAVSDK_DIR" ]; then
    echo -e "${RED}❌ MAVSDK non trouvé dans $MAVSDK_DIR${NC}"
    echo "Clonez-le avec: git clone https://github.com/mavlink/MAVSDK.git"
    exit 1
fi

# Nettoyer les processus précédents
echo -e "${YELLOW}Nettoyage des processus existants...${NC}"
pkill -9 -f mavsdk_server 2>/dev/null || true
pkill -9 -f px4 2>/dev/null || true
pkill -9 -f gazebo 2>/dev/null || true
pkill -9 -f gzserver 2>/dev/null || true
pkill -9 -f gzclient 2>/dev/null || true

# Supprimer les logs
rm -rf $LOG_DIR
sleep 3

# Créer un répertoire pour les logs
mkdir -p $LOG_DIR
echo -e "${GREEN}Logs seront sauvegardés dans: $LOG_DIR${NC}"

# Tableau pour stocker les PIDs
declare -a PX4_PIDS
declare -a MAVSDK_PIDS

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
    
    # Tuer Gazebo, px4 orphelins et mavsdk_server restants
    pkill -9 -f mavsdk_server 2>/dev/null || true
    pkill -9 -f "bin/px4" 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f gazebo 2>/dev/null || true
    pkill -9 -f gzserver 2>/dev/null || true
    pkill -9 -f gzclient 2>/dev/null || true
    
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

# ========== ÉTAPE 2 : Build PX4 ==========
echo -e "\n${BLUE}=========================================="
echo "Build PX4 SITL"
echo -e "==========================================${NC}"

cd $PX4_DIR

# Build PX4 SITL (sans lancer de simulation)
echo -e "${YELLOW}Build PX4 SITL en cours...${NC}"
make px4_sitl_default > $LOG_DIR/px4_build.log 2>&1
echo -e "${GREEN}✓ Build PX4 terminé${NC}"

# ========== ÉTAPE 3 : Lancer toutes les instances PX4 ==========
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

# ========== ÉTAPE 4 : Vérification ==========
echo -e "\n${BLUE}Vérification des processus...${NC}"
ALL_OK=true

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

# ========== ÉTAPE 5 : Lancer la simulation Python ==========
echo -e "\n${BLUE}=========================================="
echo "Lancement de la simulation Python"
echo -e "==========================================${NC}"

sleep 3

echo -e "${GREEN}Prêt pour le décollage!${NC}"
echo -e "${YELLOW}Appuyez sur Entrée pour lancer la simulation Python...${NC}"
read

# Lancer le script Python
source $GOTOWP_DIR/Mav/bin/activate
python3 $GOTOWP_DIR/dronePx4.py

# Nettoyage final
cleanup
