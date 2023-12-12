# TORCH.AUTOGRAD를 사용한 자동 미분
# 신경망을 학습할 때 가장 자주 사용되는 알고리즘은 역전파입니다. 이 알고리즘에서,
# 매개변수(모델 가중치)는 주어진 매개변수에 대한 손실 함수의 변화도(gradient)에 따라 조정됩니다.
import torch

x = torch.ones(5)  # 입력 텐서
y = torch.zeros(3)  # 출력 텐서
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Gradient 계산하기
loss.backward()
print(w.grad)
print(b.grad)

# Gradient 계산 중지 방법
z = torch.matmul(x, w)+b
z_det = z.detach() # detach() 함수를 이용해 gradient 계산을 멈춘다.
print(z_det.requires_grad)

