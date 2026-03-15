#!/usr/bin/env python3
"""
에피소드 디렉토리에서 result_episode.json이 없는 에피소드를 삭제하는 스크립트
"""

import os
import shutil
from pathlib import Path

def cleanup_episodes(base_paths):
    """
    주어진 경로들에서 result_episode.json이 없는 에피소드 디렉토리를 삭제
    
    Args:
        base_paths: 체크할 기본 경로들의 리스트
    """
    total_deleted = 0
    total_kept = 0
    
    for base_path in base_paths:
        base_path = Path(base_path)
        if not base_path.exists():
            print(f"⚠️  경로가 존재하지 않습니다: {base_path}")
            continue
            
        print(f"\n📁 검사 중: {base_path}")
        
        # run_* 디렉토리들을 찾기
        run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
        
        for run_dir in run_dirs:
            print(f"  📂 {run_dir.name} 검사 중...")
            deleted_count = 0
            kept_count = 0
            
            # 각 run 디렉토리 내의 에피소드 디렉토리들 체크
            episode_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            
            for episode_dir in episode_dirs:
                result_file = episode_dir / "result_episode.json"
                
                if not result_file.exists():
                    print(f"    🗑️  삭제: {episode_dir.name} (result_episode.json 없음)")
                    try:
                        shutil.rmtree(episode_dir)
                        deleted_count += 1
                    except Exception as e:
                        print(f"    ❌ 삭제 실패: {episode_dir.name} - {e}")
                else:
                    kept_count += 1
            
            print(f"    ✅ {run_dir.name}: 유지 {kept_count}개, 삭제 {deleted_count}개")
            total_deleted += deleted_count
            total_kept += kept_count
    
    print(f"\n📊 전체 결과:")
    print(f"  ✅ 유지된 에피소드: {total_kept}개")
    print(f"  🗑️  삭제된 에피소드: {total_deleted}개")

def main():
    # 정리할 기본 경로들
    base_paths = [
        "../../tdw_maco/test_data/cook_basic/train/cook",
        "../../tdw_maco/test_data/game_clockwise/train/game",
        "../../tdw_maco/train_data/cook_basic/train/cook", 
        "../../tdw_maco/train_data/game_clockwise/train/game"
    ]
    
    print("🧹 에피소드 정리 시작...")
    print("result_episode.json이 없는 에피소드 디렉토리를 삭제합니다.")
    
    # 사용자 확인
    response = input("\n계속하시겠습니까? (y/N): ").lower().strip()
    if response != 'y':
        print("취소되었습니다.")
        return
    
    cleanup_episodes(base_paths)
    print("\n✅ 정리 완료!")

if __name__ == "__main__":
    main()
