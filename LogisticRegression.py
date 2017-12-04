'''
Sigmoid函数和Logistic回归分类器
最优化理论
梯度下降最优化算法
数据中的缺失值处理

Logistic回归可以看成是一种概率估计

优点：计算代价不高，易于理解和实现
缺点：容易欠拟合，分类精度可能不高
适用数据类别：数值型和标签型数据


Sigmoid Function（Step Function）
f(x) = 1 / (1 + exp(-z))


梯度上升法：求解函数极大值（局部最大值）
不断迭代寻优直到达到某个指定值或者算法达到某个可以允许的误差范围内
#
# 伪代码
#
每个回归系数初始化为1
重复R次：
    计算整个数据集的梯度
    使用alpha*gradient更新回归系数的向量
    返回回归系数
'''

import os
os.chdir("/Users/huliangyong/Docs/MachineLearning/machinelearninginaction/Ch05")

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    with open("testSet.txt") as f:
        data_ = f.readlines()
        data = np.array(list(map(lambda x: x.strip().split(), data_)))
        dataMat = np.c_[np.ones(data.shape[0]), data[:, :2].astype(float)]
        labelMat = data[:, -1].astype(int)
    return dataMat, labelMat


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradientAscent(testData, classLabels):
    classLabels = classLabels.reshape(-1, 1)
    m, n = testData.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(testData @ weights)
        error = classLabels - h
        weights = weights + alpha * testData.T @ error
    return weights


def stochasticGradientAscent(testData, classLabels):
    m, n = testData.shape
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(testData[i] @ weights)
        error = classLabels[i] - h
        weights = weights + alpha * error * testData[i]
    return weights



def improvedStochGradAsc(testData, classLabels, numIter=150):
    m, n = testData.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = np.arange(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(testData[randIndex] @ weights)
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * testData[randIndex]
            np.delete(dataIndex, randIndex)
    return weights


def classifyVector(X, weights):
    prob = sigmoid(X @ weights)
    result = np.array([1 if s > 0.5 else 0 for s in prob])
    return result

def colicTest():
    with open("horseColicTraining.txt") as f:
        training_data = np.array(list(map(lambda x: x.strip().split('\t'), f.readlines()))).astype(float)
    with open('horseColicTest.txt') as f:
        test_data = np.array(list(map(lambda x: x.strip().split('\t'), f.readlines()))).astype(float)
    training_data_set = training_data[:, :-1]
    training_data_label = training_data[:, -1]
    test_data_set = test_data[:, :-1]
    test_data_label = test_data[:, -1]
    training_weights = improvedStochGradAsc(training_data_set, training_data_label, 500)
    predict_label = classifyVector(test_data_set, training_weights)
    diff = predict_label - test_data_label
    error_rate = len([s for s in diff if s != 0]) / len(test_data_label)
    return error_rate


def multiTest():
    import logging
    numTests = 10
    errorRateSum = 0.0
    for k in range(numTests):
        errorRate = colicTest()
        errorRateSum += errorRate
        print("The error rate is {}".format(errorRate))
    print('The average error rate is {0} after {1} iterations'.format(errorRateSum / numTests, numTests))


def plotBestFit(weights):
    testData, classLabels = loadDataSet()
    m, n = testData.shape
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(m):
        if int(classLabels[i]) == 1:
            xcord1.append(testData[i, 1])
            ycord1.append(testData[i, 2])
        else:
            xcord2.append(testData[i, 1])
            ycord2.append(testData[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='r', marker='o')
    ax.scatter(xcord2, ycord2, s=30, c='g')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # 为什么设置整个值为0？
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel("X2")
    plt.show()


# ————--------------------------------------------------------------------------
def plot_sigmoid_function():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()
    return


# 不同phi_z取值对单一示例样本分类代价图
def plot_loss_function():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(phi_z, -np.log(phi_z), ls='dotted')
    plt.plot(phi_z, -np.log(1 - phi_z))

    plt.xlabel('$\phi(z)$')
    plt.ylabel('J(w)')
    plt.title('Single instance loss function')
    return plt.show()

