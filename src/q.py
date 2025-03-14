
import os
import json
from collections import Counter

# 현재 스크립트 파일의 디렉토리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 프로젝트 루트 디렉토리

# JSON 파일이 저장된 폴더 경로 설정
json_folder = os.path.join(project_root, "data", "json")
# JSON 파일 목록 가져오기 및 정렬
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])

# 동작 클래스 개수를 저장할 카운터 초기화
activity_counter = Counter()

# 모든 JSON 파일 순회
for json_file in json_files:
    file_path = os.path.join(json_folder, json_file)
    
    # JSON 파일 로드
    with open(file_path, 'r') as f:
        segments = json.load(f)
    
    # 각 세그먼트의 activity 값 카운트
    for segment in segments:
        activity = segment["activity"]
        activity_counter[activity] += 1

# 결과 출력
print("\n동작 클래스별 개수:")
print("-" * 30)
for activity, count in sorted(activity_counter.items()):
    print(f"{activity:12s}: {count:3d}")

print("\n총 세그먼트 수:", sum(activity_counter.values()))
