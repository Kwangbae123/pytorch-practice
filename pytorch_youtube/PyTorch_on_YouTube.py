# PYTORCH 소개
# https://tutorials.pytorch.kr/beginner/introyt/introyt1_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

z = torch.zeros(5, 3) # 0으로 채워진 5x3 행렬 생성
# print(z)
# print(z.dtype) # tensor의 타입 출력

i = torch.ones((5, 3), dtype=torch.int16) # 정수형 데이터 타입으로 생성
# print(i)

torch.manual_seed(1729) # manual_seed(value) value값에 따라 동일한 난수 생성 r1 = r3
r1 = torch.rand(2, 2)
# print('랜덤 tensor 값: {0}'.format(r1))

r2 = torch.rand(2, 2)
# print('다른 랜덤 tensor 값: {0}'.format(r2))

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
# print('\nr1과 일치: {0}'.format(r3))

ones = torch.ones(2, 3)
#print(ones)

twos = torch.ones(2, 3) * 2 # 모든원소에 x2 를 한다
#print(twos)

threes = ones + twos # tensor의 shape이 같기 때문에 더할수 있다.
#print(threes) # 원소별 더한 값이 결과로 나온다.
#print(threes.shape)

r = (torch.rand(2, 2) - 0.5) * 2 # -1 ~ 1 사이 값
#print('랜덤 행렬값 :')
#print(r)
#print(torch.abs(r)) # 절댓값 출력
#print(torch.asin(r)) # arc sine함수 사용
#print(torch.det(r)) # 행렬식 계산
#print(torch.svd(r)) # 특이값 분해..?
#print(torch.std_mean(r)) # 평균및 표준편차
#print(torch.max(r)) # 최댓값

#--------------------------------------------------------------
# PyTorch Models
class LeNet(nn.Module): #모듈은 중첩이 가능하고 nn.Module에서 상속된다.
    def __init__(self):
        super(LeNet, self).__init__()
        # 1개의 입력 이미지 채널, 6개의 output 채널, 5x5 정방 합성곱 커널을 사용합니다.
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Affine 변환: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 이미지 차원
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 최대 풀링은 (2, 2) 윈도우 크기를 사용합니다.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 정방 사이즈인 경우, 단일 숫자만 지정할 수 있습니다.
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 크기는 배치 차원을 제외한 모든 차원을 가져옵니다.
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet()
#print(net) # 네트워크 구조 출력
input = torch.rand(1, 1, 32, 32)
#print(input.shape) # 32x32 크기의 1채널의 흑백 이미지 생성
output = net(input)
#print(output) # 결과값 출력
#print(output.shape) # 결과값의 형상

