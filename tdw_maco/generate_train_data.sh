#!/bin/bash

echo "=== 트레이닝 데이터 생성 (Cook + Game Clockwise 각 1000개) ==="

# 환경 변수 설정
export COMBO_DIR=$(pwd)
export PYTHONPATH=$COMBO_DIR:$PYTHONPATH

# 데이터셋 크기 설정
TRAIN_EPISODES=600
MAX_EPISODE=9999

# 랜덤 에피소드 선택 함수
generate_random_episodes() {
    local num_episodes=$1
    local max_episode=$2
    shuf -i 0-$max_episode -n $num_episodes | tr '\n' ' '
}

# 1. Cook 트레이닝 데이터 생성
echo ""
echo "🍳 Cook 기본 에이전트 트레이닝 데이터 생성"
port=12078
pkill -f -9 "port\ $port"

TRAIN_EPISODES_COOK_BASIC=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
echo "📊 사용할 에피소드 ($TRAIN_EPISODES개): $TRAIN_EPISODES_COOK_BASIC"
mkdir -p train_data/cook_basic

echo "🚀 Cook 시작: $(date)"
DISPLAY=:1 python challenge.py \
    --task cook \
    --output_dir train_data/cook_basic \
    --data_path train.json \
    --data_prefix dataset/ \
    --agents_algo cook_plan_agent cook_plan_agent \
    --eval_episodes $TRAIN_EPISODES_COOK_BASIC \
    --screen_size 336 \
    --port $port

echo "✅ Cook 완료: $(date)"
pkill -f -9 "port\ $port"

# 2. Game Clockwise 트레이닝 데이터 생성
# echo ""
# echo "🎮 Game 시계방향 에이전트 트레이닝 데이터 생성"
# port=12077
# pkill -f -9 "port\ $port"

# TRAIN_EPISODES_GAME_CLOCKWISE=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
# echo "📊 사용할 에피소드 ($TRAIN_EPISODES개): $TRAIN_EPISODES_GAME_CLOCKWISE"
# mkdir -p train_data/game_clockwise

# echo "🚀 Game 시작: $(date)"
# DISPLAY=:1 python challenge.py \
#     --task game \
#     --output_dir train_data/game_clockwise \
#     --data_path train.json \
#     --data_prefix dataset/ \
#     --agents_algo game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise \
#     --eval_episodes $TRAIN_EPISODES_GAME_CLOCKWISE \
#     --screen_size 336 \
#     --port $port

# echo "✅ Game 완료: $(date)"
# pkill -f -9 "port\ $port"

# 결과 확인
echo ""
echo "📊 트레이닝 데이터 생성 결과:"
if [ -d "train_data/cook_basic" ]; then
    cook_count=$(ls train_data/cook_basic 2>/dev/null | wc -l)
    echo "🍳 Cook 데이터: $cook_count개"
else
    echo "❌ Cook 데이터 생성 실패"
fi

# if [ -d "train_data/game_clockwise" ]; then
#     game_count=$(ls train_data/game_clockwise 2>/dev/null | wc -l)
#     echo "🎮 Game 데이터: $game_count개"
# else
#     echo "❌ Game 데이터 생성 실패"
# fi

echo "🎉 트레이닝 데이터 생성 완료: $(date)"
