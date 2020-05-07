import numpy as np
from matplotlib import pyplot as plt

#加载并处理数据
def load_datasets():
    points = np.genfromtxt("logistic_regression_data.csv", delimiter=",")
    x = points[:, 0:2]
    y = points[:, 2]
    print(x)
    print(y)
    return x,y

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def gradAscent(x, y, lr=0.0001, epochs=10):
    x = np.mat(x)
    print(x)
    y = np.mat(y).transpose()    #转置
    m, n = np.shape(x)
    weights = np.ones((n,1))
    print(weights)
    for k in range(epochs):
        h = sigmoid(x * weights)
        error = y - h
        weights += lr * x.transpose() * error
    return weights

def plot(x, y, w):
    n = np.shape(x)[0]
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(n):
        if int(y[i]) == 1:
            x1.append(x[i, 0])
            y1.append(x[i, 1])
        else:
            x2.append(x[i, 0])
            y2.append(x[i, 1])

    plt.scatter(x1, y1, s=30, c='red')
    plt.scatter(x2, y2, s=30, c='green')

    x1 = np.arange(0, 3, 0.1)
    y1 = (w[0] * x1) / w[1]
    plt.plot(x1,y1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('logistic regression')
    plt.show()

x,y = load_datasets()
w = gradAscent(x,y)
plot(x,y,w)