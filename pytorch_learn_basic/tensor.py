# 파이토치 튜토리얼
# https://tutorials.pytorch.kr/

# 텐서(TENSOR)

import torch
import numpy as np

# Tensor초기화
# 1. 데이터로 부터 직접 생성하기
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2. numpy배열로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. 다른 tensor로 부터 생성하기
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지한다.
# print("ones tensor : {0}".format(x_ones))

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어쓴다.
# print("random tensor : {0}".format(x_rand))

# 무작위(random) 또는 상수(constant) 값을 사용 tensor
shape = (2, 3) # shape은 tensor의 차원을 나타내는 tuple
rand_tensor = torch.rand(shape) # 무작위 tensor
ones_tensor = torch.ones(shape) # 1로 채워진 tensor
zeros_tensor = torch.zeros(shape) # 0으로 채워진 tensor

# print("Random Tensor : {0}".format(rand_tensor))
# print("Ones Tensor : {0}".format(ones_tensor))
# print("Zeros Tensor : {0}".format(zeros_tensor))

# 텐서의 속성
# tensor = torch.rand(3, 4)
# print("Shape of tensor : {0}".format(tensor.shape)) # tensor의 형태
# print("Datatype of tensor : {0}".format(tensor.dtype)) # tensor의 자료형(데이터 타입)
# print("Device tensor is stored on : {0}".format(tensor.device)) # tensor가 어느장치에 저장되는지

# 텐서 연산(Operation)
if torch.cuda.is_available(): # GPU사용이 가능하면 GPU텐서로 생성
    tensor = tensor.to("cuda")

data = [[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.], [13.,14.,15.,16.]]
tensor = torch.tensor(data)
# print("tensor row : {0}".format(tensor[0])) # tensor[0]은 tensor의 첫번째 행을 의미
# print("tensor column : {0}".format(tensor[:, 0])) # tensor[:,0]은 tensor의 첫번째 열을 의미
# print("last column : {0}".format(tensor[...,-1]))# tensor[..., -1]은 tensor의 마지막 열을 의미

# 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim= 1)
# print(t1)

# 산술 연산(Arithmetic operations)
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
# matmul함수는 스칼라와 배열의 곱 X , 3원 이상 다차원 배열곱에서 np.dot과 차이가 난다.
y1 = tensor @ tensor.T # .T는 tensor의 전치를 의미한다.
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
# print(torch.matmul(tensor, tensor.T, out=y3))

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
# print(torch.mul(tensor, tensor, out=z3))

agg = tensor.sum() #텐서내 모든요수 더하기
# print(agg)
agg_item = agg.item() # item 함수를 사용해 숫자값으로 반환
# print(agg_item, type(agg_item))

# 텐서를 NumPy 배열로 변환하기
t = torch.ones(5)
print("t : {0}".format(t))
n = t.numpy()
print("n : {0}".format(n))

# 텐서의 변경 사항이 NumPy 배열에 반영된다.
t.add_(1)
print("t : {0}".format(t))
print("n : {0}".format(n))

# NumPy 배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)

# umPy 배열의 변경 사항이 텐서에 반영된다.
np.add(n, 1, out=n)
print("t : {0}".format(t))
print("n : {0}".format(n))