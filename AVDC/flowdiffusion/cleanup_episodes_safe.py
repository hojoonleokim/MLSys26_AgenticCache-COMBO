#!/usr/bin/env python3
"""
에피소드 디렉토리에서 result_episode.json이 없는 에피소드를 삭제하는 안전한 스크립트
"""

import os
import shutil
import argparse
from pathlib import Path

def find_episodes_to_delete(base_paths, dry_run=True):
    """
    삭제할 에피소드들을 찾고, dry_run이 False면 실제로 삭제
    
    Args:
        base_paths: 체크할 기본 경로들의 리스트
        dry_run: True면 삭제하지 않고 리스트만 출력
    """
    to_delete = []
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
            
            # 각 run 디렉토리 내의 에피소드 디렉토리들 체크
            episode_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            
            for episode_dir in episode_dirs:
                result_file = episode_dir / "result_episode.json"
                
                if not result_file.exists():
                    to_delete.append(episode_dir)
                    print(f"    🔍 삭제 대상: {episode_dir}")
                else:
                    total_kept += 1
    
    print(f"\n📊 검사 결과:")
    print(f"  ✅ 유지될 에피소드: {total_kept}개")
    print(f"  🗑️  삭제 대상 에피소드: {len(to_delete)}개")
    
    if dry_run:
        print(f"\n🔍 DRY RUN 모드 - 실제로 삭제하지 않습니다")
        if to_delete:
            print("삭제 대상 목록:")
            for episode_dir in to_delete:
                print(f"  - {episode_dir}")
        return to_delete
    
    # 실제 삭제 수행
    if to_delete:
        print(f"\n🗑️  {len(to_delete)}개 에피소드 삭제 중...")
        deleted_count = 0
        
        for episode_dir in to_delete:
            try:
                shutil.rmtree(episode_dir)
                print(f"    ✅ 삭제 완료: {episode_dir}")
                deleted_count += 1
            except Exception as e:
                print(f"    ❌ 삭제 실패: {episode_dir} - {e}")
        
        print(f"\n✅ 삭제 완료: {deleted_count}/{len(to_delete)}개")
    else:
        print("\n✅ 삭제할 에피소드가 없습니다.")
    
    return to_delete

def main():
    parser = argparse.ArgumentParser(description="result_episode.json이 없는 에피소드 디렉토리 정리")
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='실제로 삭제하지 않고 삭제 대상만 표시 (기본값)')
    parser.add_argument('--delete', action='store_true',
                       help='실제로 삭제 수행')
    parser.add_argument('--paths', nargs='*', 
                       help='정리할 경로들 (기본값: 미리 정의된 경로들)')
    
    args = parser.parse_args()
    
    # 기본 경로들
    default_paths = [
        "../../tdw_maco/test_data/cook_basic/train/cook",
        "../../tdw_maco/test_data/game_clockwise/train/game",
        "../../tdw_maco/train_data/cook_basic/train/cook", 
        "../../tdw_maco/train_data/game_clockwise/train/game"
    ]
    
    paths = args.paths if args.paths else default_paths
    dry_run = not args.delete
    
    print("🧹 에피소드 정리 도구")
    print("result_episode.json이 없는 에피소드 디렉토리를 찾습니다.")
    print(f"모드: {'DRY RUN (삭제하지 않음)' if dry_run else '실제 삭제'}")
    
    if not dry_run:
        response = input("\n⚠️  실제로 삭제하시겠습니까? (y/N): ").lower().strip()
        if response != 'y':
            print("취소되었습니다.")
            return
    
    find_episodes_to_delete(paths, dry_run=dry_run)
    
    if dry_run:
        print("\n💡 실제로 삭제하려면: python cleanup_episodes_safe.py --delete")

if __name__ == "__main__":
    main()
