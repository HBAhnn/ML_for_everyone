import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
w1 = torch.zeros(1, requires_grad=True)
# 이 부분을 채워넣으세요.# #가중치 w1를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요.
w2 = torch.zeros(1, requires_grad=True)
# 이 부분을 채워넣으세요.# #가중치 w2를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요.
w3 = torch.zeros(1, requires_grad=True)
# 이 부분을 채워넣으세요.# #가중치 w3를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요.
b = torch.zeros(1, requires_grad=True)
# 이 부분을 채워넣으세요.# #편향 b를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요.
# optimizer 설정
optimizer =  optim.SGD([w1, w2, w3, b], lr=1e-5)
# 이 부분을 채워넣으세요.# #SGD optimizer를 사용하고 learning rate는 1e-5로 적용하세요.

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = w1 * x1_train + w2 * x2_train + w3 * x3_train + b # 이 부분을 채워넣으세요.# # H(x)=w1*x1 + w2*x2 + w3*x3 + b를 이용하세요.

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    # 이 부분을 채워넣으세요.# #파이토치 코드 상으로 선형 회귀의 비용 함수에 해당되는 평균 제곱 오차를 선언하세요.

    # cost로 H(x) 개선
    # 이 부분을 채워넣으세요.#  # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 이 부분을 채워넣으세요.#  # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # 이 부분을 채워넣으세요.#  # W와 b를 업데이트
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w3.item(), w3.item(), b.item(), cost.item()
        ))