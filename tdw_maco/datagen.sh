#!/bin/bash

echo "=== MACO 모델 데이터 생성 및 학습 스크립트 ==="
echo ""
echo "💡 데이터셋 설정:"
echo "   - TRAIN_EPISODES: 훈련용 에피소드 수 (현재: 200개)"
echo "   - TEST_EPISODES: 테스트용 에피소드 수 (현재: 50개)"
echo "   - 다양한 에이전트 조합으로 데이터 생성:"
echo "     * Cook: basic, altruism, selfish, mixed"
echo "     * Game: clockwise, counter_clockwise"
echo "   - 해상도: 128x128 (모든 모델이 동적 리사이즈로 사용)"
echo "   - 전체 사용하려면: TRAIN_EPISODES=40000, TEST_EPISODES=40000"
echo ""

# 환경 변수 설정
export COMBO_DIR=$(pwd)
export PYTHONPATH=$COMBO_DIR:$PYTHONPATH

# 데이터셋 크기 설정 (일부 에피소드만 사용)
TRAIN_EPISODES=200  # 훈련용 에피소드 수
TEST_EPISODES=50    # 테스트용 에피소드 수
MAX_EPISODE=9999     # 전체 데이터셋의 최대 에피소드 번호 (실제 데이터셋 크기에 맞게 조정)

# 랜덤 에피소드 선택 함수
generate_random_episodes() {
    local num_episodes=$1
    local max_episode=$2
    shuf -i 0-$max_episode -n $num_episodes | tr '\n' ' '
}

# 1. 학습 데이터 생성 - 다양한 에이전트 조합 (각각 다른 랜덤 에피소드 사용)
echo "=== 학습 데이터 생성 (각 데이터셋마다 랜덤 에피소드 $TRAIN_EPISODES개 선택) ==="

# 1-a. Cook 작업 - 다양한 에이전트 조합 (128x128)
echo "1-a-1. Cook 기본 에이전트 (cook_plan_agent)"
TRAIN_EPISODES_COOK_BASIC=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TRAIN_EPISODES_COOK_BASIC"
mkdir -p train_data/cook_basic
DISPLAY=:1 python challenge.py --task cook --output_dir train_data/cook_basic --data_path train.json --data_prefix dataset/ --agents_algo cook_plan_agent cook_plan_agent --eval_episodes $TRAIN_EPISODES_COOK_BASIC --screen_size 128

echo "1-a-2. Cook 이타적 에이전트 (cook_plan_agent_altruism)"
TRAIN_EPISODES_COOK_ALTRUISM=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TRAIN_EPISODES_COOK_ALTRUISM"
mkdir -p train_data/cook_altruism
DISPLAY=:1 python challenge.py --task cook --output_dir train_data/cook_altruism --data_path train.json --data_prefix dataset/ --agents_algo cook_plan_agent_altruism cook_plan_agent_altruism --eval_episodes $TRAIN_EPISODES_COOK_ALTRUISM --screen_size 128

echo "1-a-3. Cook 이기적 에이전트 (cook_plan_agent_selfish)"
TRAIN_EPISODES_COOK_SELFISH=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TRAIN_EPISODES_COOK_SELFISH"
mkdir -p train_data/cook_selfish
DISPLAY=:1 python challenge.py --task cook --output_dir train_data/cook_selfish --data_path train.json --data_prefix dataset/ --agents_algo cook_plan_agent_selfish cook_plan_agent_selfish --eval_episodes $TRAIN_EPISODES_COOK_SELFISH --screen_size 128

echo "1-a-4. Cook 혼합 에이전트 (altruism + selfish)"
TRAIN_EPISODES_COOK_MIXED=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TRAIN_EPISODES_COOK_MIXED"
mkdir -p train_data/cook_mixed
DISPLAY=:1 python challenge.py --task cook --output_dir train_data/cook_mixed --data_path train.json --data_prefix dataset/ --agents_algo cook_plan_agent_altruism cook_plan_agent_selfish --eval_episodes $TRAIN_EPISODES_COOK_MIXED --screen_size 128

# 1-b. Game 작업 - 다양한 에이전트 조합 (128x128)
echo "1-b-1. Game 시계방향 에이전트 (game_plan_agent_clockwise)"
TRAIN_EPISODES_GAME_CLOCKWISE=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TRAIN_EPISODES_GAME_CLOCKWISE"
mkdir -p train_data/game_clockwise
DISPLAY=:1 python challenge.py --task game --output_dir train_data/game_clockwise --data_path train.json --data_prefix dataset/ --agents_algo game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise --eval_episodes $TRAIN_EPISODES_GAME_CLOCKWISE --screen_size 128

echo "1-b-2. Game 반시계방향 에이전트 (game_plan_agent_counter_clockwise)"
TRAIN_EPISODES_GAME_COUNTER=$(generate_random_episodes $TRAIN_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TRAIN_EPISODES_GAME_COUNTER"
mkdir -p train_data/game_counter_clockwise
DISPLAY=:1 python challenge.py --task game --output_dir train_data/game_counter_clockwise --data_path train.json --data_prefix dataset/ --agents_algo game_plan_agent_counter_clockwise game_plan_agent_counter_clockwise game_plan_agent_counter_clockwise game_plan_agent_counter_clockwise --eval_episodes $TRAIN_EPISODES_GAME_COUNTER --screen_size 128

echo "학습 데이터 생성 완료!"
echo ""

# 2. 테스트 데이터 생성 - 다양한 에이전트 조합 (각각 다른 랜덤 에피소드 사용)
echo "=== 테스트 데이터 생성 (각 데이터셋마다 랜덤 에피소드 $TEST_EPISODES개 선택) ==="

# 2-a. Cook 작업 - 다양한 에이전트 조합 (128x128)
echo "2-a-1. Cook 기본 에이전트 테스트"
TEST_EPISODES_COOK_BASIC=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TEST_EPISODES_COOK_BASIC"
mkdir -p test_data/cook_basic
DISPLAY=:1 python challenge.py --task cook --output_dir test_data/cook_basic --data_path test.json --data_prefix dataset/ --agents_algo cook_plan_agent cook_plan_agent --eval_episodes $TEST_EPISODES_COOK_BASIC --screen_size 128

echo "2-a-2. Cook 이타적 에이전트 테스트"
TEST_EPISODES_COOK_ALTRUISM=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TEST_EPISODES_COOK_ALTRUISM"
mkdir -p test_data/cook_altruism
DISPLAY=:1 python challenge.py --task cook --output_dir test_data/cook_altruism --data_path test.json --data_prefix dataset/ --agents_algo cook_plan_agent_altruism cook_plan_agent_altruism --eval_episodes $TEST_EPISODES_COOK_ALTRUISM --screen_size 128

echo "2-a-3. Cook 이기적 에이전트 테스트"
TEST_EPISODES_COOK_SELFISH=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TEST_EPISODES_COOK_SELFISH"
mkdir -p test_data/cook_selfish
DISPLAY=:1 python challenge.py --task cook --output_dir test_data/cook_selfish --data_path test.json --data_prefix dataset/ --agents_algo cook_plan_agent_selfish cook_plan_agent_selfish --eval_episodes $TEST_EPISODES_COOK_SELFISH --screen_size 128

echo "2-a-4. Cook 혼합 에이전트 테스트"
TEST_EPISODES_COOK_MIXED=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TEST_EPISODES_COOK_MIXED"
mkdir -p test_data/cook_mixed
DISPLAY=:1 python challenge.py --task cook --output_dir test_data/cook_mixed --data_path test.json --data_prefix dataset/ --agents_algo cook_plan_agent_altruism cook_plan_agent_selfish --eval_episodes $TEST_EPISODES_COOK_MIXED --screen_size 128

# 2-b. Game 작업 - 다양한 에이전트 조합 (128x128)
echo "2-b-1. Game 시계방향 에이전트 테스트"
TEST_EPISODES_GAME_CLOCKWISE=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TEST_EPISODES_GAME_CLOCKWISE"
mkdir -p test_data/game_clockwise
DISPLAY=:1 python challenge.py --task game --output_dir test_data/game_clockwise --data_path test.json --data_prefix dataset/ --agents_algo game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise game_plan_agent_clockwise --eval_episodes $TEST_EPISODES_GAME_CLOCKWISE --screen_size 128

echo "2-b-2. Game 반시계방향 에이전트 테스트"
TEST_EPISODES_GAME_COUNTER=$(generate_random_episodes $TEST_EPISODES $MAX_EPISODE)
echo "  사용할 에피소드: $TEST_EPISODES_GAME_COUNTER"
mkdir -p test_data/game_counter_clockwise
DISPLAY=:1 python challenge.py --task game --output_dir test_data/game_counter_clockwise --data_path test.json --data_prefix dataset/ --agents_algo game_plan_agent_counter_clockwise game_plan_agent_counter_clockwise game_plan_agent_counter_clockwise game_plan_agent_counter_clockwise --eval_episodes $TEST_EPISODES_GAME_COUNTER --screen_size 128

echo "테스트 데이터 생성 완료!"
echo ""

# # 3. AVDC 디렉토리로 이동
# cd AVDC/flowdiffusion

# # 4. 모델별 전처리 및 학습
# echo "=== 3개 Diffusion 모델 학습 시작 ==="
# echo ""

# # 4-1. Multiple 모델 (기본 모델) - 모든 에이전트 조합 사용
# echo "4-1. Multiple 모델 전처리 및 학습..."
# echo "전처리 중..."
# python train_maco.py -m preprocess --task_name cook_basic cook_altruism cook_selfish cook_mixed game_clockwise game_counter_clockwise
# if [ $? -eq 0 ]; then
#     echo "Multiple 모델 전처리 완료"
#     echo "Multiple 모델 학습 시작..."
#     python train_maco.py -m train --task_name cook_basic cook_altruism cook_selfish cook_mixed game_clockwise game_counter_clockwise --save_milestone
#     echo "Multiple 모델 학습 완료"
#     echo "모델 저장 위치: results/tdw_maco_multiple"
# else
#     echo "Multiple 모델 전처리 실패"
# fi
# echo ""

# # 4-2. Single 모델 - 모든 에이전트 조합 사용
# echo "4-2. Single 모델 전처리 및 학습..."
# echo "전처리 중..."
# python train_maco.py -m preprocess --single --task_name cook_basic cook_altruism cook_selfish cook_mixed game_clockwise game_counter_clockwise
# if [ $? -eq 0 ]; then
#     echo "Single 모델 전처리 완료"
#     echo "Single 모델 학습 시작..."
#     python train_maco.py -m train --single --task_name cook_basic cook_altruism cook_selfish cook_mixed game_clockwise game_counter_clockwise --save_milestone
#     echo "Single 모델 학습 완료"
#     echo "모델 저장 위치: results/tdw_maco_single"
# else
#     echo "Single 모델 전처리 실패"
# fi
# echo ""

# # 4-3. Inpainting 모델 - 모든 에이전트 조합 사용
# echo "4-3. Inpainting 모델 전처리 및 학습..."
# echo "전처리 중..."
# python train_maco.py -m preprocess --inpainting --task_name cook_basic cook_altruism cook_selfish cook_mixed game_clockwise game_counter_clockwise
# if [ $? -eq 0 ]; then
#     echo "Inpainting 모델 전처리 완료"
#     echo "Inpainting 모델 학습 시작..."
#     python train_maco.py -m train --inpainting --task_name cook_basic cook_altruism cook_selfish cook_mixed game_clockwise game_counter_clockwise --save_milestone
#     echo "Inpainting 모델 학습 완료"
#     echo "모델 저장 위치: results/tdw_maco_inpainting"
# else
#     echo "Inpainting 모델 전처리 실패"
# fi
# echo ""

# # 4-4. Super Resolution 모델 - 모든 에이전트 조합 데이터 사용
# echo "4-4. Super Resolution 모델 학습..."
# echo "사용할 데이터:"
# echo "  - 훈련: train_data/ 의 모든 하위 디렉토리 (6가지 에이전트 조합)"
# echo "  - 테스트: test_data/ 의 모든 하위 디렉토리 (6가지 에이전트 조합)"
# echo "Super Resolution 모델 학습 시작..."
# python train_super_res.py -m train --save_milestone
# if [ $? -eq 0 ]; then
#     echo "Super Resolution 모델 학습 완료"
#     echo "모델 저장 위치: results/super_res"
#     echo "모델 파일 생성 중..."
#     python train_super_res.py -m get_model
#     echo "Super Resolution 모델 파일 생성 완료: results/super_res/super_res_model.pt"
# else
#     echo "Super Resolution 모델 학습 실패"
# fi
# echo ""

# # 5. 원래 디렉토리로 복귀
# cd $COMBO_DIR

# echo "=== 전체 과정 완료! ==="
# echo "생성된 모델들:"
# echo "1. Multiple 모델: AVDC/flowdiffusion/results/tdw_maco_multiple"
# echo "2. Single 모델: AVDC/flowdiffusion/results/tdw_maco_single"
# echo "3. Inpainting 모델: AVDC/flowdiffusion/results/tdw_maco_inpainting"
# echo "4. Super Resolution 모델: AVDC/flowdiffusion/results/super_res"
# echo ""
# echo "CWM에서 사용할 모델 파일들:"
# echo "- VDM 모델: results/tdw_maco_multiple/model-*.pt 또는 results/tdw_maco_single/model-*.pt"
# echo "- Inpainting 모델: results/tdw_maco_inpainting/model-*.pt"
# echo "- Super Resolution 모델: results/super_res/super_res_model.pt"
# echo ""
# echo "각 모델의 체크포인트와 샘플 이미지가 해당 디렉토리에 저장됩니다."