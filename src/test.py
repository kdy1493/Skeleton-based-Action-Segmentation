# PyTorch 라이브러리 임포트 - 딥러닝 모델 구현 및 실행에 사용
import torch
# NumPy 라이브러리 임포트 - 배열 처리 및 수치 연산에 사용
import numpy as np
# JSON 파일 처리를 위한 라이브러리 임포트
import json
# 운영체제 관련 기능(파일 경로 등)을 위한 라이브러리 임포트
import os
# 시스템 관련 기능을 위한 라이브러리 임포트
import sys

# 현재 스크립트 파일의 디렉토리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 경로 계산 (현재 디렉토리의 상위 디렉토리)
project_root = os.path.dirname(current_dir)

# 프로젝트 루트 디렉토리를 Python 모듈 검색 경로에 추가 (net 모듈을 찾기 위함)
sys.path.append(project_root)

# ST-GCN 모델 클래스 임포트
from net.st_gcn import Model

def load_model(model_path=None, in_channels=2, num_class=5):
    """
    ST-GCN 모델을 로드하는 함수
    
    Args:
        model_path (str, optional): 모델 가중치 파일 경로
        in_channels (int, optional): 입력 채널 수
        num_class (int, optional): 출력 클래스 수
    
    Returns:
        tuple: (model, device) - 로드된 모델과 사용 장치
    """
    # ST-GCN 모델 인스턴스 생성
    model = Model(
        in_channels=in_channels,
        num_class=num_class,
        graph_args={'layout': 'kinetics', 'strategy': 'spatial'},
        edge_importance_weighting=True
    )
    
    # 모델 가중치 파일 경로 설정
    if model_path is None:
        model_path = os.path.join(project_root, "work_dir", "custom", "epoch50_model.pt")
    
    # 저장된 모델 가중치 로드
    model.load_state_dict(torch.load(model_path))
    # 모델을 평가 모드로 설정
    model.eval()
    
    # GPU 사용 가능 여부 확인 후 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델을 지정된 장치로 이동
    model = model.to(device)
    
    return model, device

def load_data(data_path=None, label_path=None):
    """
    데이터와 레이블을 로드하는 함수
    
    Args:
        data_path (str, optional): 데이터 파일 경로
        label_path (str, optional): 레이블 파일 경로
    
    Returns:
        tuple: (all_data, all_labels) - 로드된 데이터와 레이블
    """
    # 경로가 지정되지 않은 경우 기본 경로 사용
    if data_path is None:
        data_path = os.path.join(project_root, "data", "custom_train_data.npy")
    if label_path is None:
        label_path = os.path.join(project_root, "data", "custom_train_label.npy")
    
    # 데이터와 레이블 로드
    all_data = np.load(data_path)
    all_labels = np.load(label_path)
    
    return all_data, all_labels

def get_action_maps():
    """
    동작 클래스 매핑 딕셔너리를 반환하는 함수
    
    Returns:
        tuple: (action_map, action_map_rev) - 정방향 및 역방향 매핑 딕셔너리
    """
    # 동작 클래스 번호와 이름 매핑 딕셔너리 (숫자 -> 문자열)
    action_map = {0: "no_activity", 1: "standing", 2: "sitting", 3: "walking", 4: "no_presence"}
    # 역방향 매핑 딕셔너리 (문자열 -> 숫자)
    action_map_rev = {"no_activity": 0, "standing": 1, "sitting": 2, "walking": 3, "no_presence": 4}
    
    return action_map, action_map_rev

def predict_segment(model, segment_data, device, min_frames=10):
    """
    구간 데이터에 대한 예측을 수행하는 함수
    
    Args:
        model: ST-GCN 모델
        segment_data (np.ndarray): 구간 데이터
        device: 사용 장치 (CPU 또는 GPU)
        min_frames (int, optional): 최소 프레임 수
    
    Returns:
        tuple: (pred, pred_probs) - 예측 클래스와 클래스별 확률
    """
    # 구간이 최소 프레임 수보다 짧으면 패딩 추가
    if segment_data.shape[1] < min_frames:
        # 필요한 패딩 길이 계산
        pad_length = min_frames - segment_data.shape[1]
        # 패딩 생성 (0으로 채움)
        padding = np.zeros((segment_data.shape[0], pad_length, segment_data.shape[2], segment_data.shape[3]))
        # 원본 데이터와 패딩 연결
        segment_data = np.concatenate([segment_data, padding], axis=1)
    
    # 배치 차원 추가 (모델 입력 형태 맞추기)
    segment_data = np.expand_dims(segment_data, 0)
    # NumPy 배열을 PyTorch 텐서로 변환하고 지정된 장치로 이동
    segment_tensor = torch.tensor(segment_data, dtype=torch.float32).to(device)
    
    # 모델 예측 수행 (그래디언트 계산 없이)
    with torch.no_grad():
        # 모델에 데이터 입력하여 출력 얻기
        output = model(segment_tensor)
        # 출력을 확률로 변환 (소프트맥스 적용)
        pred_probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        # 가장 높은 확률을 가진 클래스 인덱스 가져오기
        pred = np.argmax(pred_probs)
    
    return pred, pred_probs

def analyze_sequence(model, sequence_data, json_file, device, action_map, action_map_rev, sequence_idx):
    """
    시퀀스 데이터를 분석하는 함수
    
    Args:
        model: ST-GCN 모델
        sequence_data (np.ndarray): 시퀀스 데이터
        json_file (str): JSON 파일명
        device: 사용 장치 (CPU 또는 GPU)
        action_map: 정방향 매핑 딕셔너리 (숫자 -> 문자열)
        action_map_rev: 역방향 매핑 딕셔너리 (문자열 -> 숫자)
        sequence_idx (int): 시퀀스 인덱스
    
    Returns:
        tuple: (segment_results, sequence_pred) - 구간별 결과와 전체 시퀀스 예측
    """
    # 시퀀스 분석 시작 구분선 출력
    print(f"\n{'='*70}")
    print(f"시퀀스 {sequence_idx} 분석 - {json_file}")
    print(f"{'='*70}")
    
    # JSON 파일 로드
    with open(os.path.join(project_root, "data", "json", json_file), 'r') as f:
        segments = json.load(f)
    
    segment_results = []
    
    # 각 구간(세그먼트)에 대해 반복
    for segment in segments:
        # 구간의 시작 및 끝 프레임 인덱스 가져오기
        start, end = segment["frameRange"]
        # 구간의 실제 동작 레이블 가져오기
        true_action = segment["activity"]
        # 문자열 레이블을 숫자로 변환
        true_label = action_map_rev.get(true_action, 0)
        
        # 구간 데이터 추출
        segment_data = sequence_data[:, start:end + 1, :, :]
        
        # 구간 예측 수행
        pred, pred_probs = predict_segment(model, segment_data, device)
        # 예측된 클래스 인덱스를 동작 이름으로 변환
        pred_action = action_map[pred]
        
        # 예측 결과 출력 (정확히 예측했는지 여부 표시)
        match = "✓" if pred_action == true_action else "✗"
        print(f"프레임 [{start:3d}, {end:3d}] ({end-start+1:3d}프레임): 실제: {true_action:12s}, 예측: {pred_action:12s} {match}")
        # 각 클래스별 예측 확률 출력
        print(f"  확률: {' '.join([f'{action_map[i]}:{prob:.2f}' for i, prob in enumerate(pred_probs)])}")
        
        # 결과 저장
        segment_results.append({
            "start": start,
            "end": end,
            "true_action": true_action,
            "true_label": true_label,
            "pred_action": pred_action,
            "pred": pred,
            "pred_probs": pred_probs,
            "correct": pred_action == true_action
        })
    
    # 전체 시퀀스에 대한 예측 수행
    print("\n전체 시퀀스 예측:")
    # 전체 시퀀스 예측
    sequence_pred, sequence_probs = predict_segment(model, sequence_data, device, min_frames=1)
    # 예측된 클래스 인덱스를 동작 이름으로 변환
    sequence_pred_action = action_map[sequence_pred]
    
    return segment_results, (sequence_pred, sequence_probs)

def calculate_statistics(all_results, all_sequence_preds, all_labels, action_map):
    """
    분석 결과에 대한 통계를 계산하는 함수
    
    Args:
        all_results (list): 모든 구간 분석 결과
        all_sequence_preds (list): 모든 시퀀스 예측 결과
        all_labels (np.ndarray): 실제 시퀀스 레이블
        action_map: 정방향 매핑 딕셔너리 (숫자 -> 문자열)
    
    Returns:
        dict: 계산된 통계 정보
    """
    # 전체 정확도 통계를 위한 변수 초기화
    total_correct = 0  # 정확히 예측한 구간 수
    total_segments = 0  # 전체 구간 수
    # 클래스별 정확도 통계를 위한 딕셔너리 초기화
    class_correct = {i: 0 for i in range(5)}  # 각 클래스별 정확히 예측한 수
    class_total = {i: 0 for i in range(5)}    # 각 클래스별 전체 수
    
    # 모든 구간 결과에 대해 통계 계산
    for results in all_results:
        for result in results:
            total_segments += 1
            if result["correct"]:
                total_correct += 1
            
            class_total[result["true_label"]] += 1
            if result["pred"] == result["true_label"]:
                class_correct[result["true_label"]] += 1
    
    # 시퀀스 정확도 계산
    sequence_correct = 0
    for i, (pred, _) in enumerate(all_sequence_preds):
        if pred == all_labels[i]:
            sequence_correct += 1
    
    # 통계 정보 반환
    return {
        "segment_accuracy": total_correct / total_segments if total_segments > 0 else 0,
        "total_correct": total_correct,
        "total_segments": total_segments,
        "class_correct": class_correct,
        "class_total": class_total,
        "sequence_correct": sequence_correct,
        "total_sequences": len(all_labels),
        "sequence_accuracy": sequence_correct / len(all_labels) if len(all_labels) > 0 else 0
    }

def print_statistics(stats, action_map):
    """
    통계 정보를 출력하는 함수
    
    Args:
        stats (dict): 계산된 통계 정보
        action_map: 정방향 매핑 딕셔너리 (숫자 -> 문자열)
    """
    print("\n" + "="*70)
    print("최종 통계:")
    print("="*70)
    # 구간별 정확도 출력
    print(f"구간별 정확도: {stats['segment_accuracy']:.2%} ({stats['total_correct']}/{stats['total_segments']})")
    
    # 클래스별 정확도 출력
    print("\n클래스별 정확도:")
    for i in range(5):
        if stats['class_total'][i] > 0:
            accuracy = stats['class_correct'][i] / stats['class_total'][i]
            print(f"{action_map[i]:12s}: {accuracy:.2%} ({stats['class_correct'][i]}/{stats['class_total'][i]})")
        else:
            print(f"{action_map[i]:12s}: 데이터 없음")
    
    # 전체 시퀀스 정확도 출력
    print(f"\n전체 시퀀스 정확도: {stats['sequence_accuracy']:.2%} ({stats['sequence_correct']}/{stats['total_sequences']})")

def test_model(model_path=None, data_path=None, label_path=None, json_folder=None):
    """
    모델 테스트를 수행하는 메인 함수
    
    Args:
        model_path (str, optional): 모델 가중치 파일 경로
        data_path (str, optional): 데이터 파일 경로
        label_path (str, optional): 레이블 파일 경로
        json_folder (str, optional): JSON 파일 폴더 경로
    """
    # 모델 로드
    model, device = load_model(model_path)
    
    # 데이터 로드
    all_data, all_labels = load_data(data_path, label_path)
    
    # 동작 매핑 가져오기
    action_map, action_map_rev = get_action_maps()
    
    # JSON 파일 목록 가져오기
    if json_folder is None:
        json_folder = os.path.join(project_root, "data", "json")
    json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])
    
    all_results = []
    all_sequence_preds = []
    
    # 모든 시퀀스에 대해 반복
    for sequence_idx in range(len(all_data)):
        # 현재 시퀀스 데이터 가져오기
        sequence_data = all_data[sequence_idx]
        
        # 현재 시퀀스에 대한 JSON 파일이 있는지 확인
        if sequence_idx < len(json_files):
            # 현재 시퀀스의 JSON 파일명 가져오기
            json_file = json_files[sequence_idx]
            
            # 시퀀스 분석
            segment_results, sequence_pred = analyze_sequence(
                model, sequence_data, json_file, device, action_map, action_map_rev, sequence_idx
            )
            
            all_results.append(segment_results)
            all_sequence_preds.append(sequence_pred)
            
            # 전체 시퀀스의 실제 레이블 가져오기
            true_label = all_labels[sequence_idx]
            # 실제 레이블을 동작 이름으로 변환
            true_action = action_map[true_label]
            # 정확히 예측했는지 여부 표시
            match = "✓" if action_map[sequence_pred[0]] == true_action else "✗"
            # 전체 시퀀스 예측 결과 출력
            print(f"전체 시퀀스: 실제: {true_action:12s}, 예측: {action_map[sequence_pred[0]]:12s} {match}")
            # 각 클래스별 예측 확률 출력
            print(f"  확률: {' '.join([f'{action_map[i]}:{prob:.2f}' for i, prob in enumerate(sequence_pred[1])])}")
        else:
            # JSON 파일이 없는 경우 메시지 출력
            print(f"\n시퀀스 {sequence_idx}에 대한 JSON 파일이 없습니다.")
    
    # 통계 계산
    stats = calculate_statistics(all_results, all_sequence_preds, all_labels, action_map)
    
    # 통계 출력
    print_statistics(stats, action_map)

# 스크립트로 직접 실행될 때만 함수 호출
if __name__ == "__main__":
    test_model()