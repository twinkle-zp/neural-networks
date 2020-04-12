import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#加载并处理数据
def load_datasets():
    points = np.genfromtxt("data.csv", delimiter=",")
    x = points[:, 0]
    y = points[:, 1]
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x,y

#对y=wx+b做梯度下降
def gradient_descent(x,y,w,b,lr):
    grad_b = 0
    grad_w = 0
    num = float(len(x))
    for i in range(0,x.shape[0]):
        x_in = x[i]    #取出一组数据
        y_out = y[i]
        # 对b求梯度grad_b = 2(wx+b-y)
        grad_b += (2/num)*((w*x_in+b)-y_out)
        # 对w求梯度grad_w = 2(wx+b-y)*x
        grad_w += (2/num)*x_in*((w*x_in +b)-y_out)
    # 更新参数
    new_b = b-(lr*grad_b)
    new_w = w-(lr*grad_w)
    return new_w,new_b

#计算均方误差
def loss(x,y,w,b):
    error = 0  #总误差
    mean_squared_error = 0  #均方误差
    for i in range(0,x.shape[0]):
        x1 = x[i]
        y1 = y[i]
        # mean-squared-error(均方误差)平方和
        error += (y1-(w*x1+b))**2
    # 每个点的平均loss
    mean_squared_error = error/float(len(x))
    return mean_squared_error

#画图
def plot(x_np,y_np,w,b):
    y = w * x_np + b
    plt.title("linear regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_np, y)
    plt.scatter(x_np, y_np)
    plt.show()

#参数初始化
lr = 0.000001
b = 0
w = 0
epochs = 90

x,y = load_datasets()

#数据集可视化
plt.title("dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x.numpy(), y.numpy())
plt.show()

print(x.shape[0])
print("训练前loss：{0}".format(loss(x,y,w,b)))
for i in range(epochs):
    w,b = gradient_descent(x,y,w,b,lr)
print("训练后loss：{0}".format(loss(x,y,w,b)))
print("训练后w：{0},b：{1}".format(w,b))
plot(x.numpy(),y.numpy(),w,b);

