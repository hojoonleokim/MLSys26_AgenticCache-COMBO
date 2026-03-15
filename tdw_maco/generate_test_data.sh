#!/bin/bash

echo "=== 테스트 데이터 생성 (Cook + Game Clockwise 각 200개) ==="

# 환경 변수 설정
export COMBO_DIR=$(pwd)
export PYTHONPATH=$COMBO_DIR:$PYTHONPATH

# 데이터셋 크기 설정
TEST_EPISODES=50
MAX_EPISODE=9999

# 랜덤 에피소드 선택 함수
generate_random_episodes() {
    local num_episodes=$1
    local max_episode=$2
    shuf -i 0-$max_episode -n $num_episodes | tr '\n' ' '
}

# 1. Cook 테스트 데이터 생성
echo ""
echo "🍳 Cook 기본 에이전트 테스트 데이터 생성"
port=12078
pkill -f -9 "port\ $port"

TEST_EPISODES_COOK_BASIC=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "📊 사용할 에피소드 ($TEST_EPISODES개): $TEST_EPISODES_COOK_BASIC"
mkdir -p test_data/cook_basic

echo "🚀 Cook 시작: $(date)"
DISPLAY=:1 python challenge.py \
    --task cook \
    --output_dir test_data/cook_basic \
    --data_path test.json \
    --data_prefix dataset/ \
    --agents_algo cook_plan_agent cook_plan_agent \
    --eval_episodes $TEST_EPISODES_COOK_BASIC \
    --screen_size 336 \
    --port $port

echo "✅ Cook 완료: $(date)"
pkill -f -9 "port\ $port"

# 2. Game Clockwise 테스트 데이터 생성
echo ""
echo "🎮 Game 시계방향 에이전트 테스트 데이터 생성"
port=12077
pkill -f -9 "port\ $port"

TEST_EPISODES_GAME_CLOCKWISE=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "📊 사용할 에피소드 ($TEST_EPISODES개): $TEST_EPISODES_GAME_CLOCKWISE"
mkdir -p test_data/game_clockwise

echo "🚀 Game 시작: $(date)"
DISPLAY=:1 python challenge.py \
    --task game \
    --output_dir test_data/game_clockwise \
    --data_path test.json \
    --data_prefix dataset/ \
    --agents_algo game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise \
    --eval_episodes $TEST_EPISODES_GAME_CLOCKWISE \
    --screen_size 336 \
    --port $port

echo "✅ Game 완료: $(date)"
pkill -f -9 "port\ $port"

# 결과 확인
echo ""
echo "📊 생성 결과:"
if [ -d "test_data/cook_basic" ]; then
    cook_count=$(ls test_data/cook_basic 2>/dev/null | wc -l)
    echo "🍳 Cook 데이터: $cook_count개"
else
    echo "❌ Cook 데이터 생성 실패"
fi

if [ -d "test_data/game_clockwise" ]; then
    game_count=$(ls test_data/game_clockwise 2>/dev/null | wc -l)
    echo "🎮 Game 데이터: $game_count개"
else
    echo "❌ Game 데이터 생성 실패"
fi

echo "🎉 전체 완료: $(date)"
