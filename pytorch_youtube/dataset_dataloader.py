# Datasets 및 Dataloaders
import torch
import torchvision
import torchvision.transforms as transforms

# transforms.ToTensor() 불러온 이미지를 tensor형태로 변환한다.
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# https://tutorials.pytorch.kr/beginner/introyt/introyt1_tutorial.html 여가까지
