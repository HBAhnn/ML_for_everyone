import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5) # 이 부분을 채워넣으세요.# #SGD optimizer를 사용하고 learning rate는 1e-5로 적용하세요.
 
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # 이 부분을 채워넣으세요.# #PyTorch에서 기본적으로 제공하는 mse 함수를 사용하세요.

    # cost로 H(x) 개선
    # 이 부분을 채워넣으세요.#  # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 이 부분을 채워넣으세요.#  # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # 이 부분을 채워넣으세요.#  # W와 b를 업데이트
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))