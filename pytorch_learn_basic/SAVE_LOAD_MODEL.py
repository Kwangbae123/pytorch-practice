# 모델 저장하고 불러오기
import sys, os
sys.path.append(os.pardir)
import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth') # torch.save 메소드를 사용하여 저장(persist)할수 있다.

model = models.vgg16() # 여기서는 `weights`를 지정하지 않았으므로, 학습되지 않은 모델을 생성합니다.
model.load_state_dict(torch.load('model_weights.pth')) # 모델 가중치를 불러오기 위해서 load_state_dict() 메소드를 사용하여 매개변수들을 불러온다.
print(model.eval())

# 이 클래스의 구조를 모델과 함께 저장 model 을 저장 함수에 전달합니다
torch.save(model, 'model.pth')
model = torch.load('model.pth')