import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
W = torch.zeros([3,1], requires_grad=True) # 이 부분을 채워넣으세요.# #가중치 W를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요. x_train의 shape을 잘 생각해보세요!!
b = torch.zeros(1, requires_grad=True) # 이 부분을 채워넣으세요.# #가중치 W를 0으로 초기화하고 학습을 통해 값이 변경될 수 있도록 하세요.
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5) # 이 부분을 채워넣으세요.# #SGD optimizer를 사용하고 learning rate는 1e-5로 적용하세요.

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = x_train.matmul(W) + b  # or .mm or @ #x_train의 shape이 달라졌기 때문에 행렬곱셉을 합니다!!

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2) # 이 부분을 채워넣으세요.# #파이토치 코드 상으로 선형 회귀의 비용 함수에 해당되는 평균 제곱 오차를 선언하세요.

    # cost로 H(x) 개선
    # 이 부분을 채워넣으세요.#  # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 이 부분을 채워넣으세요.#  # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # 이 부분을 채워넣으세요.#  # W와 b를 업데이트
    optimizer.step()

    # 100번마다 로그 출력
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))