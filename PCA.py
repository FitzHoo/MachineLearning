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

plt.bar(range(1, len(eigen_vals)+1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(eigen_vals)+1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# mapping to lower dimensions
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))   # 将行数组转变为列形式
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
lr.fit(X_train_pca, y_train)   # 训练得到模型的参数
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

# Step 2: 计算类内散步矩阵
num = len(np.unique(y))
d = X_train_std.shape[1]
S_W = np.zeros((d, d))
for label, mv in zip(range(num), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X[y == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: {} x {}'.format(S_W.shape[0], S_W.shape[1]))

