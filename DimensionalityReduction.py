# PCA建模步骤
# 1) 对原始d维数据集做标准化处理
# 2) 构造样本的协方差矩阵
# 3) 计算协方差矩阵的特征值和相应的特征变量
# 4) 选择与前k个最大特征值对应的特征向量，其中k为新特征空间的维度(k<=d)
# 5) 通过前k个特征向量构建映射矩阵W
# 6) 通过映射矩阵W将d维的输入数据集X转换到新的k维特征子空间

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

wine_data = load_wine()
wine_df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
wine_df['target'] = wine_data['target']

from sklearn.model_selection import train_test_split

X, y = wine_df.iloc[:, :-1].values, wine_df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Step 1:Standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

# Step 2: Construct Covariance Matrix
cov_mat = np.cov(X_train_std.T)

# Step 3: Calculate Eigenvalues And Eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Variance Explained Ratios
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, len(eigen_vals) + 1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(eigen_vals) + 1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# mapping to lower dimensions
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))  # 将行数组转变为列形式
print('Matrix W:\n', w)

# Transfer the original data(124 X 13) to pca space(124 X 2)
X_train_pca = X_train_std @ w

# Visualization
colors = ['r', 'g', 'b']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(loc='best')
plt.show()

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.2):
    fig = plt.figure()
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper left')

    return plt.show()


# 先对原始数据降维，然后基于降维后的数据进行逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr.fit(X_train_pca, y_train)  # 训练得到模型的参数
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plot_decision_regions(X_test_pca, y_test, classifier=lr)

# LDA建模步骤
# 前提：特征呈正态分布且特征间相互独立

# 1) 对d维数据集进行标准化处理（d为特征的数量)
# 2) 对于每一类别，计算d维的均值向量
# 3) 构造类间的散布矩阵S_{B}已经类内的散布矩阵S_{W}
# 4) 计算矩阵S_{W}^{-1} * S_{B}的特征值及对应的特征向量
# 5) 选取前k个特征值所对应的特征向量，构造一个d x k维的转换矩阵W，其中特征向量以列的形式排列
# 6) 使用转换矩阵W将样本映射到新的特征子空间上

# Step 1: 计算类内均值向量
np.set_printoptions(precision=3)
mean_vecs = []
for label in range(3):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('mean vector {}: \n {} \n '.format(label, mean_vecs[label]))

# Step 2: 计算类内散布矩阵
num = len(np.unique(y))
d = X_train_std.shape[1]
# S_W = np.zeros((d, d))
# for label, mv in zip(range(num), mean_vecs):
#     class_scatter = np.zeros((d, d))
#     for row in X[y == label]:
#         row, mv = row.reshape(d, 1), mv.reshape(d, 1)
#         class_scatter += (row - mv).dot((row - mv).T)
#     S_W += class_scatter
# print('Within-class scatter matrix: {} x {}'.format(S_W.shape[0], S_W.shape[1]))

S_W = np.zeros((d, d))
for label, mv in zip(range(num), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label], rowvar=False)  # np.cov默认对row进行转化，而非column
    S_W += class_scatter
print('Within-class scatter matrix: {} x {}'.format(S_W.shape[0], S_W.shape[1]))

# Step 3: 计算类间散布矩阵
mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X[y == i, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall) @ (mean_vec - mean_overall).T
print('Between-class scatter matrix: {} x {}'.format(S_B.shape[0], S_B.shape[1]))

# Step 4: 计算广义特征值
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

# Variance Explained Ratios
tot = sum(eigen_vals)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, len(eigen_vals) + 1), discr, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(eigen_vals) + 1), cum_discr, where='mid', label='cumulative explained variance')
plt.ylim([-0.1, 1.1])
plt.ylabel('Discriminability Ratio')
plt.xlabel('Linear Discriminants')
plt.show()

# Step 5: 利用判别能力最强的特征向量构建转换矩阵W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

# Step 6: 通过转换矩阵将样本映射到新的特征空间
# Variance Explained Ratios
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, len(eigen_vals) + 1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(eigen_vals) + 1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='best')
plt.show()

# ----------------------------------------------------------------------
# 使用Python实现核主成分分析
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh


def rbf_kernel_pca(X, gamma, n_components):
    '''
    RBF kernel PCA implementation
    :param X: ndarray, shape = [n_samples, n_features]
    :param gamma: float, tuning parameter of the RBF kernel
    :param n_components: int, number of pricinpal components to return
    :return: ndarray, shape = [n_samples, n_features]
        X_pc, projected dataset
    '''
    # Calculate pairwise squared Euclidean distances in the M * N dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances in to a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)  # 使核矩阵更为聚集

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alpha = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]

    return alpha, lambdas


def plot_raw_data(X, y):
    fig = plt.figure()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
    return plt.show()


def plot_linear_pca(X, y, n_components=2):
    scikit_pca = PCA(n_components=n_components)
    X_spca = scikit_pca.fit_transform(X)
    n = X_spca.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y == 0, 0], np.zeros((n//2, 1)) + 0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y == 1, 0], np.zeros((n//2, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC 1')
    ax[0].set_ylabel("PC 2")
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    return plt.show()


def plot_kernel_pca(X, y, gamma=15, n_components=2):
    from matplotlib.ticker import FormatStrFormatter
    X_kpca = rbf_kernel_pca(X, gamma=gamma, n_components=n_components)
    n = X_kpca.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y == 0, 0], np.zeros((n//2, 1)) + 0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y == 1, 0], np.zeros((n//2, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC 1')
    ax[0].set_ylabel("PC 2")
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    return plt.show()


# Separate Half-Moon/Semilune
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)
plot_raw_data(X, y)
plot_linear_pca(X, y)
plot_kernel_pca(X, y)


# Separate concentric circles
from sklearn.datasets import make_circles
X_, y_ = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plot_raw_data(X_, y_)
plot_linear_pca(X_, y_)
plot_kernel_pca(X_, y_)


# kernal pca in scikit learn
from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
fig = plt.figure()
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()
