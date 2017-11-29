# 序列最小优化(Squential Minimal Optimization, SMO)算法

# 支持向量指离分隔超平面最近的那些点
# 找到最小间隔的数据点，并使其与超平面的间隔最大化
# 几乎所有的分类问题都可以使用SVM，SVM本身是一个二元分类器
# 核函数(kernel)将数据从一个低维空间映射到一个高维空间，可以将一个在低维空间中的非线性问题转换成高维空间下的线性问题来求解



import os

os.chdir("/Users/huliangyong/Docs/MachineLearning/machinelearninginaction/Ch06")
import numpy as np


def loadDataSet(filename):
    with open(filename) as f:
        data_ = f.readlines()
        data = np.array(list(map(lambda x: x.strip().split(), data_)))
        dataMat = np.c_[np.ones(data.shape[0]), data[:, :2].astype(float)]
        labelMat = data[:, -1].astype(int)
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


'''
创建一个alpha向量并将其初始化为0向量
当迭代次数小于最大迭代次数时（外循环）
    对数据集中的每个数据向量（内循环）：
        如果该数据向量可以被优化：
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
如果所有向量都没被优化，增加迭代数目，继续下一次循环
'''


class optStruct:
    def __init__(self, test_data, test_label, C, toler):
        self.X = test_data
        self.label = test_label
        self.C = C
        self.tol = toler
        self.m = test_data.shape[0]  # 数据个数
        self.alphas = np.zeros((self.m, 1))  # 每个数据点对应一个alpha值
        self.b = 0
        self.err_cache = np.zeros((self.m, 2))  # 误差缓存

    def calc_ek(self, k):
        fxk = np.float(np.multiply(self.alphas, self.label).T * (self.X * self.X[k, :].T)) + self.b
        ek = fxk - np.float(self.label[k])
        return ek

    def select_j(self, i, ei):
        max_k = -1
        max_delta_e = 0
        ej = 0
        self.err_cache[i] = [1, ei]  # 内循环中的启发式方法
        valid_err_cache_list = np.nonzero(self.err_cache[:, 0])[0]
        if len(valid_err_cache_list) > 1:
            for k in valid_err_cache_list:
                if k == i:
                    continue
                ek = self.calc_ek(k)
                delta_e = np.abs(ei - ek)
                if delta_e > max_delta_e:
                    max_k = k
                    max_delta_e = delta_e
                    ej = ek
        return max_k, ej

    def update_ek(self, k):
        ek = self.calc_ek(k)
        self.err_cache[k] = [1, ek]


# --------------------------------------------------------------
class SVMSMO:
    '''
    SEQUENTIAL MINIMAL OPTIMIZATION APPLIED TO SUPPORT VECTOR MACHINE
    '''

    def __int__(self, max_iter=1000, kernel_type='linear', C=1.0, epsilon=0.001, sigma=5.0):
        '''

        :param max_iter: maximum iteration
        :param kernel_type: kernel type to use in training.
                            'linear' represents linear kernel function
                            'quadratic' represents quadratic kernel function
                            'gaussian' represents gaussian kernel function
        :param C: value of regularization parameter C
        :param epsilon: convergence value
        :param sigma: parameter for gaussian kernel
        :return:
        '''

        self.kernels = {'linear': self.kernel_linear,
                        'quadratic': self.kernel_quadratic,
                        'gaussain': self.kernel_gaussian}
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.sigma = sigma

    def fit(self, X, y):
        # Initialization
        n, d = X.shape
        alpha = np.zeros((n))
        # kernel = self.kernels[self.kernel_type]
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n - 1, j)  # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_i, alpha_prime_j = alpha[i], alpha[j]
                L, H = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j)) / k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of {} iterations".format(self.max_iter))
                return

        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)

        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(alpha * y, X)

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if y_i != y_j:
            return max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_j + alpha_prime_j)
        else:
            return max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_j + alpha_prime_j)

    def get_rnd_int(self, a, b, z):
        i = z
        while i == z:
            i = np.random.randint(a, b)
        return i

    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return np.dot(x1, x2.T) ** 2

    def kernel_gaussian(self, x1, x2, sigma=5.0):
        if self.sigma:
            sigma = self.sigma
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / 2 * (sigma ** 2))


