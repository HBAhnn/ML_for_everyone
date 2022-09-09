import torch
import torch.nn as nn

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = torch.FloatTensor([[0], [1], [1], [0]])

model = nn.Sequential(
          nn.Linear(2, 10, bias=True),
          nn.Sigmoid(),
          nn.Linear(10, 10, bias=True),
          nn.Sigmoid(),
          nn.Linear(10, 10, bias=True),
          nn.Sigmoid(),
          nn.Linear(10, 1, bias=True),
          nn.Sigmoid()
          )

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(),
          '\nAccuracy: ', accuracy.item())

