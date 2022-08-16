import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


#hyper parameters
training_epochs = 10
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', #데이터를 다운로드 받을 경로
                          train=True, #True는 train 데이터
                          transform=transforms.ToTensor(), #데이터를 PyTorch 텐서로 변환
                          download=True) #해당 경로에 MNIST 데이터가 없다면 다운로드

#test 데이터셋.
mnist_test = dsets.MNIST(root='MNIST_data/',#이 부분을 채워넣으세요.#, #데이터를 다운로드 받을 경로 : 'MNIST_data/'
                         train=False,#이 부분을 채워넣으세요.#, #False는 test 데이터
                         transform=transforms.ToTensor(),#이 부분을 채워넣으세요.#,
                         download=True)#이 부분을 채워넣으세요.#)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, #로드할 대상
                                          batch_size=batch_size, #배치크기
                                          shuffle=True, #Epoch마다 데이터 셋을 섞어서 데이터가 학습되는 순서를 바꿈.
                                          drop_last=True) #마지막 배치를 버릴 것인지

# 입력 784 : 데이터 이미지의 전체 픽셀 / 출력 10 : class(label)의 갯수
linear = torch.nn.Linear(784, 10, bias=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

#Training
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28)
        Y = Y

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))


#Test
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다. test 할때는 gradient 계산이 필요가 없다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float()
    Y_test = mnist_test.test_labels

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test # 각각의 데이터 마다 가장 큰 값의 인덱스를 취함 => class를 예측하는 것
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float()
    Y_single_data = mnist_test.test_labels[r:r + 1]

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()