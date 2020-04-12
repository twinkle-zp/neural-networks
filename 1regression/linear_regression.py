import numpy as np
from matplotlib import pyplot as plt
# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]   #第i行第0列
        y = points[i, 1]   #第i行第1列
        # mean-squared-error(均方误差)平方和
        totalError += (y - (w * x + b)) ** 2
    # 每个点的平均loss
    return totalError / float(len(points))

#更新梯度（即更新w,b）
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 对b求梯度grad_b = 2(wx+b-y)
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        # 对w求梯度grad_w = 2(wx+b-y)*x
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
    # 更新参数
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # 训练轮数
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]

#画图
def plot(x_np,y_np,w,b):
    y = w * x_np + b
    plt.title("linear regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_np, y)
    plt.scatter(x_np, y_np)
    plt.show()

def run():
	
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_w = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )
    x=points[:,0]
    y=points[:,1]
    plot(x,y,w,b)

if __name__ == '__main__':
    run()