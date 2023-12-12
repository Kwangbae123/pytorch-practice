# DATASET과 DATALOADER

import sys, os
sys.path.append(os.pardir)
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader

# root 는 학습/테스트 데이터가 저장되는 경로
# train 은 학습용 또는 테스트용 데이터셋 ( True : 학습용 , False : 테스트용)
# download=True 는 root 에 데이터가 없는 경우 인터넷에서 다운로드한다.
# transform 과 target_transform 은 특징(feature)과 정답(label) 변형(transform)을 지정
training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

labels_map = { 0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot",}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # # 1차원 배열로 training_dataset 길이만큼 랜덤 정수 1개 생성
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# 사용자 정의 Dataset 클래스는 __init__, __len__, __getitem__ 함수 3가지가 구현되어있어햐 한다.
class CustomImageDataset(Dataset):
    # __init__ 함수는 Dataset 객체가 생성(instantiate)될 때 한 번만 실행된다.
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # __len__ 함수는 데이터셋의 샘플 개수를 반환합니다.
    def __len__(self):
        return len(self.img_labels)

    # __getitem__ 함수는 주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환한다.
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# DataLoader로 학습용 데이터 준비하기
# 미니배치(minibatch)로 전달, 매 epoch마다 데이터를 섞어서 과적합을 방지한다.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print("Feature batch shape: {0}".format(train_features.size()))
print("Labels batch shape: {0}".format(train_labels.size()))
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print("Label : {0}".format(label))



