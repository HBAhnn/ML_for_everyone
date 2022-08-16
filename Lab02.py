import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1, requires_grad=True)

b = torch.zeros(1, requires_grad=True)

hypothesis = x_train * W + b

cost = torch.mean((hypothesis - y_train) ** 2)

optimizer = optim.SGD([W, b], lr=0.01)

optimizer.zero_grad()
cost.backward()
optimizer.step()

hypothesis = x_train * W + b

cost = torch.mean((hypothesis - y_train) ** 2)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
#가중치 W를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요.
b = torch.zeros(1, requires_grad=True)
#편향 b를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요.
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)
#SGD optimizer를 사용하고 learning rate는 0.01로 적용하세요.

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    #파이토치 코드 상으로 선형 회귀의 비용 함수에 해당되는 평균 제곱 오차를 선언하세요.
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))