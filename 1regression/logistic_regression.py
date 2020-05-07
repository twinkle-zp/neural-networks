from math import exp
from numpy import *
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升算法
    :param dataMatIn: 数据集，一个二维Numpy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
    :param classLabels: 类别标签，是一个（1，100）的行向量。
    :return: 返回回归系数
    '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    # m->样本数，n->特征数 在本例中m=100,n=3
    m, n = shape(dataMatrix)
    # alpha代表向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 权重值初始全为1
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # shape = (100,1)
        error = labelMat - h  # shape = (100,1)
        weights += alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent(dataMatIn, classLabels, numIter=150):
    '''
    随机梯度上升法
    :param dataMatIn: 数据集，一个二维Numpy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
    :param classLabels: 类别标签，是一个（1，100）的行向量。
    :param numIter: 外循环次数
    :return: 返回回归系数
    '''
    dataMatrix = mat(dataMatIn)
    m, n = shape(dataMatrix)
    weights = ones((n, 1))
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 随机产生一个 0～len(dataIndex)之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(random.uniform(0, len(dataIndex)))
            choose = dataIndex[randIndex]
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=w1*x1+w2*x2+..+wn*xn
            h = sigmoid(sum(dataMatrix[choose] * weights))
            error = classLabels[choose] - h
            weights += alpha * error * dataMatrix[choose].transpose()
            del (dataIndex[randIndex])
    return weights


def showData(dataArr, labelMat):
    '''
    展示数据集分布情况
    :param dataArr:
    :param labelMat:
    :return: None
    '''
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    p1 = ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    p2 = ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.legend([p1, p2], ['Class 1', 'Class 0'], loc='lower right', scatterpoints=1)
    plt.show()


def plotBestFit(dataArr, labelMat, weights1, weights2):
    '''
    将我们得到的数据可视化展示出来
    :param dataArr:样本数据的特征
    :param labelMat:样本数据的类别标签，即目标变量
    :param weights:回归系数
    :return:None
    '''
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax1.scatter(xcord2, ycord2, s=30, c='green')
    ax2.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax2.scatter(xcord2, ycord2, s=30, c='green')
    x1 = arange(-3.0, 3.0, 0.1)
    y1 = (-weights1[0] - weights1[1] * x1) / weights1[2]
    x2 = arange(-3.0, 3.0, 0.1)
    y2 = (-weights2[0] - weights2[1] * x2) / weights2[2]
    ax1.plot(x1, y1)
    ax2.plot(x2, y2)
    ax1_title_text = ax1.set_title(u'梯度上升算法', FontProperties=font)
    ax2_title_text = ax2.set_title(u'随机梯度上升算法', FontProperties=font)
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.setp(ax1_title_text, size=20, weight='bold', color='black')
    # plt.setp(ax2_title_text, size=20, weight='bold', color='black')
    plt.show()


def testLR():
    dataMat, classLabels = loadDataSet('data/TestSet.txt')
    dataArr = array(dataMat)
    weights1 = gradAscent(dataArr, classLabels)
    weights2 = stocGradAscent(dataArr, classLabels)
    test(dataArr, classLabels)
    plotBestFit(dataArr, classLabels, weights1, weights2)


if __name__ == '__main__':
    testLR()