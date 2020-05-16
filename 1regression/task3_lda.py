import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

# 数据预处理
def preprocess():
    # 使用sklearn中包随机生成二分类数据
    x, y = datasets.samples_generator.make_classification(n_samples=400, n_features=2, n_redundant=0, n_classes=2,
                                                          n_informative=1, n_clusters_per_class=1)
    # 取出两类数据
    x1 = []
    x2 = []
    for i in range(len(x)):
        if y[i] == 0:
            x1.append(x[i])
        else:
            x2.append(x[i])

    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

    return x,y,x1,x2

# 线性类别分析LDA
def LDA(x1, x2, y):

    len1 = len(x1)
    len2 = len(x2)

    # 求均值
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)

    # 计算Sw
    cov1 = np.dot(np.transpose(x1 - mean1), (x1 - mean1))
    cov2 = np.dot(np.transpose(x2 - mean2), (x2 - mean2))
    Sw = cov1 + cov2

    # 计算w
    w = np.dot(np.mat(Sw).I, (mean1 - mean2).reshape((len(mean1), 1)))  # 计算w
    x1 = np.dot((x1), w)
    x2 = np.dot((x2), w)
    y1 = np.ones(len1)
    y2 = np.ones(len2)

    return x1, x2, y1, y2


if '__main__' == __name__:

    x, y, x1, x2 = preprocess()

    x1, x2, y1, y2 = LDA(x1 ,x2, y)

    plt.plot(x1, y1, 'bo')
    plt.plot(x2, y2, 'ro')
    plt.show()
