# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap=True):
        # 기존 코드 주석 처리
        # with open(self.data_path, 'rb') as f:
        #     self.sample_name, self.label = pickle.load(f)
        
        # 커스텀 데이터 로드 (numpy 배열)
        try:
            # numpy 배열로 직접 로드
            self.data = np.load(self.data_path)
            self.label = np.load(self.label_path)
            
            # 샘플 이름 생성 (필요한 경우)
            self.sample_name = [str(i) for i in range(len(self.label))]
            
            print(f"데이터 로드 성공: {self.data.shape}, 레이블: {self.label.shape}")
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            raise

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def get_data(self, index):
        # 기존 코드 주석 처리
        # data_numpy = np.load(self.data_path + '/' + self.sample_name[index] + '.npy')
        # label = self.label[index]
        
        # 커스텀 데이터에서 직접 가져오기
        data_numpy = self.data[index]
        label = self.label[index]
        
        return data_numpy, label