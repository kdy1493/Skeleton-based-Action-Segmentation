#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ST-GCN 모델을 사용한 동작 예측 스크립트
사용법: python predict.py
"""

import os
import sys

# 현재 스크립트 파일의 디렉토리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# test.py의 test_model 함수 임포트
from src.test import test_model

if __name__ == "__main__":
    # 인자 없이 test_model 함수 호출
    test_model()