import os
import pandas as pd

def clean_csv_headers(csv_folder=None):
    """
    CSV 파일의 두 번째 행을 삭제하는 함수
    
    Args:
        csv_folder (str, optional): CSV 파일이 저장된 폴더 경로. 
                                   None인 경우 기본 경로(../data/csv)를 사용합니다.
    
    Returns:
        None
    """
    # 폴더 경로가 지정되지 않은 경우 기본 경로 사용
    if csv_folder is None:
        # 현재 스크립트 파일의 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # CSV 파일들이 저장된 폴더 경로 (data/csv 폴더)
        csv_folder = os.path.join(current_dir, "..", "data", "csv")
    
    # 파일 리스트 가져오기
    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    
    for file in csv_files:
        file_path = os.path.join(csv_folder, file)
        
        # CSV 파일을 텍스트로 읽기
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 두 번째 행(인덱스 1) 삭제
        if len(lines) > 1:
            new_lines = [lines[0]] + lines[2:]  # 첫 번째 행과 세 번째 행 이후만 유지
            
            # 변경된 내용을 파일에 쓰기
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            print(f"{file}: 두 번째 행이 삭제되었습니다.")
        else:
            print(f"{file}: 행이 충분하지 않습니다.")
    
    print("✅ 처리 완료!")

# 스크립트로 직접 실행될 때만 함수 호출
if __name__ == "__main__":
    clean_csv_headers()