# 신경망 모델 구성하기
# 신경망은 데이터에 대한 연산을 수행하는 계층(layer)/모듈(module)로 구성
# torch.nn : 신경망을 구성하는데 필요한 모든 구성 요소를 제공
#
import os, sys
sys.path.append(os.pardir)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 학습을 위한 장치 설정
device = ( "cuda" if torch.cuda.is_available() else "mps"  if torch.backends.mps.is_available() else "cpu")
print("Using {0} device".format(device))

class NeuralNetwork(nn.Module):
    # 신경망 계층 초기화
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 네트워크의 구조 출력
model = NeuralNetwork().to(device)
# print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X) # 모델에 입력을 전달하여 호출하면 2차원 텐서를 반환
pred_probab = nn.Softmax(dim=1)(logits) # dim = 0  row를 기준으로 softmax계산 / dim = 1 colunm을 기준으로 softmax계산
y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# FashionMNIST 모델의 계층
# 28x28 크기의 이미지 3개로 구성된 미니배치를 가져온다.
input_image = torch.rand(3,28,28)
print(input_image.size())

# nn.Flatten() 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear 선형 계층 은 저장된 가중치(weight)와 편향(bias)을 사용하여 입력에 선형 변환(linear transformation)을 적용
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#순차 컨테이너(sequential container)를 사용하여 신경망을 빠르게 만들 수 있습니다.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax : 모델의 각 분류(class)에 대한 예측 확률을 나타내도록 [0, 1] 범위로 비례하여 조정(scale)된다.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)