import pandas as pd
import numpy as np
import os
import json

def load_csv_data(csv_folder=None):
    """
    CSV 파일에서 데이터를 로드하는 함수
    
    Args:
        csv_folder (str, optional): CSV 파일이 저장된 폴더 경로
        
    Returns:
        np.ndarray: 로드된 데이터 배열 (N, C, T, V, M) 형태
    """
    # 현재 스크립트 파일의 디렉토리 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 폴더 경로 설정
    if csv_folder is None:
        csv_folder = os.path.join(current_dir, "..", "data", "csv")
    
    # 파일 목록 가져오기
    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    
    # 데이터 처리
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(file_path)
        seq = df.drop(columns=["frame"]).values.reshape((102, -1, 2))
        seq = np.transpose(seq, (2, 0, 1))  # (C, T, V)
        all_data.append(seq)
    
    all_data = np.array(all_data)  # (N, C, T, V)
    all_data = np.expand_dims(all_data, axis=-1)  # (N, C, T, V, M), M=1
    
    print("all_data shape:", all_data.shape)
    return all_data

def load_json_labels(json_folder=None, n_samples=None, n_frames=102):
    """
    JSON 파일에서 레이블을 로드하는 함수
    
    Args:
        json_folder (str, optional): JSON 파일이 저장된 폴더 경로
        n_samples (int, optional): 처리할 샘플 수
        n_frames (int, optional): 각 시퀀스의 프레임 수
        
    Returns:
        np.ndarray: 로드된 레이블 배열 (N, T) 형태
    """
    # 현재 스크립트 파일의 디렉토리 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 폴더 경로 설정
    if json_folder is None:
        json_folder = os.path.join(current_dir, "..", "data", "json")
    
    # 파일 목록 가져오기
    json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])
    
    if n_samples is None:
        n_samples = len(json_files)
    else:
        n_samples = min(n_samples, len(json_files))
    
    # 동작 레이블 매핑
    action_map = {"no_activity": 0, "standing": 1, "sitting": 2, "walking": 3, "no_presence": 4}
    
    # 프레임별 레이블 배열 (N, T)
    all_labels = np.zeros((n_samples, n_frames), dtype=int)
    
    # 각 json 파일 처리
    for i, json_file in enumerate(json_files[:n_samples]):
        file_path = os.path.join(json_folder, json_file)
        with open(file_path, 'r') as f:
            segments = json.load(f)
            for segment in segments:
                start, end = segment["frameRange"]
                action = action_map[segment["activity"]]
                all_labels[i, start:end + 1] = action  # 끝 프레임 포함
    
    print("All labels shape:", all_labels.shape)
    print("Sample labels for sequence 0:", all_labels[0])
    return all_labels

def split_train_test(data, labels, train_ratio=0.8, random_seed=None):
    """
    데이터와 레이블을 훈련 세트와 테스트 세트로 분할하는 함수
    
    Args:
        data (np.ndarray): 입력 데이터
        labels (np.ndarray): 레이블 데이터
        train_ratio (float, optional): 훈련 데이터 비율
        random_seed (int, optional): 랜덤 시드 값
        
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels) 형태의 튜플
    """
    # 랜덤 시드 설정
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    
    # 인덱스 섞기 (랜덤 분할)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # 훈련 데이터와 레이블
    train_data = data[train_indices]
    # 각 시퀀스에서 가장 많이 등장하는 레이블을 해당 시퀀스의 레이블로 사용
    train_labels = np.array([np.bincount(labels[i]).argmax() for i in train_indices])
    
    # 테스트 데이터와 레이블
    test_data = data[test_indices]
    test_labels = np.array([np.bincount(labels[i]).argmax() for i in test_indices])
    
    return train_data, train_labels, test_data, test_labels

def save_dataset(train_data, train_labels, test_data, test_labels, save_dir=None):
    """
    데이터셋을 파일로 저장하는 함수
    
    Args:
        train_data (np.ndarray): 훈련 데이터
        train_labels (np.ndarray): 훈련 레이블
        test_data (np.ndarray): 테스트 데이터
        test_labels (np.ndarray): 테스트 레이블
        save_dir (str, optional): 저장 경로
    """
    # 현재 스크립트 파일의 디렉토리 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 저장 경로 설정
    if save_dir is None:
        save_dir = os.path.join(current_dir, "..", "data")
    
    # 데이터 저장
    np.save(os.path.join(save_dir, "custom_train_data.npy"), train_data)
    np.save(os.path.join(save_dir, "custom_train_label.npy"), train_labels)
    np.save(os.path.join(save_dir, "custom_test_data.npy"), test_data)
    np.save(os.path.join(save_dir, "custom_test_label.npy"), test_labels)
    
    print(f"훈련 데이터 저장 완료: {train_data.shape}, 레이블: {train_labels.shape}")
    print(f"테스트 데이터 저장 완료: {test_data.shape}, 레이블: {test_labels.shape}")

def create_custom_dataset(csv_folder=None, json_folder=None, save_dir=None, train_ratio=0.8, random_seed=None):
    """
    CSV 파일과 JSON 파일에서 데이터를 읽어 커스텀 데이터셋을 생성하고 저장하는 함수
    
    Args:
        csv_folder (str, optional): CSV 파일이 저장된 폴더 경로
        json_folder (str, optional): JSON 파일이 저장된 폴더 경로
        save_dir (str, optional): 생성된 데이터셋을 저장할 경로
        train_ratio (float, optional): 훈련 데이터 비율 (기본값: 0.8)
        random_seed (int, optional): 랜덤 시드 값
    
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels) 형태의 튜플
    """
    # 데이터 로드
    data = load_csv_data(csv_folder)
    labels = load_json_labels(json_folder, n_samples=len(data))
    
    # 데이터 분할
    train_data, train_labels, test_data, test_labels = split_train_test(
        data, labels, train_ratio, random_seed
    )
    
    # 데이터 저장
    save_dataset(train_data, train_labels, test_data, test_labels, save_dir)
    
    return train_data, train_labels, test_data, test_labels

# 스크립트로 직접 실행될 때만 함수 호출
if __name__ == "__main__":
    create_custom_dataset()