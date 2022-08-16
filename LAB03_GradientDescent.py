#Gradient Descent

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = 0
#가중치 W를 0으로 초기화하세요. Gradient Descent by Hand이기 때문에 학습을 통해 값이 변경되는 것이 아닙니다!
# learning rate 설정
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = W * x_train
    # 이 부분을 채워넣으세요.# # H(x)=Wx를 이용하세요.

    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    #파이토치 코드 상으로 선형 회귀의 비용 함수에 해당되는 평균 제곱 오차를 선언하세요.
    gradient = torch.sum((W * x_train - y_train) * x_train)
    # #Gradient Descent by Hand를 이용하세요.

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, W, cost.item()
    ))

    # cost gradient로 H(x) 개선
    W -= lr * gradient
    #W:=W−α∇W을 이용하세요.