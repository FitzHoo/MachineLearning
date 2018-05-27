import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
boston = load_boston()

data = boston['data']
target = boston['target']
feature_names = boston['feature_names']

df = pd.DataFrame(data, columns=feature_names)
print(df.head())

# Exploratory Data Analysis
# Seaborn pairplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')

cols = ['LSTAT', 'INDUS', 'NOX', 'RM']
sns.pairplot(df[cols], size=2.5)
plt.show()

# 相关系数矩阵是将数据标准化后得到的协方差矩阵
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    annot_kws={'size': 15},
    yticklabels=cols,
    xticklabels=cols
)

plt.show()

class LinearRegressionGD:

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        y = y.reshape((-1, ))    # 这一步很关键
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1: ] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        output = np.dot(X, self.w_[1:]) + self.w_[0]
        return output

    def predict(self, X):
        return self.net_input(X)

# 为了使梯度下降算法收敛性更佳，对相关变量做标准化处理
X = df[['RM']].values
y = target

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1, 1))   # fit_transform 只接受矩阵形式的数据？

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# 将代价看作是迭代次数的函数（基于训练数据集）
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.xlabel('Epoch')
plt.ylabel('SSE')
plt.show()


def lin_reg_plot(X, y, model):
    plt.scatter(X, y, c='blue', alpha=0.5)
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel("Average number of rooms[RM] (Standardized)")
    plt.ylabel('Price in $1000\'s [MEDV] (Standardized)')
    return plt.show()

lin_reg_plot(X_std, y_std, lr)

# 预测带有五个房间的房屋价格
num_rooms_std = sc_x.transform(np.array([5.0]).reshape(-1, 1))  # transform expected 2D array
price_std = lr.predict(num_rooms_std)
print("Price in $1000\'s: {:.3f}".format(sc_y.inverse_transform(price_std)[0]))

# 对于经过标准化处理的变量，无需更新其截距的权重，因为它们在y轴上的截距始终为0
print('Slope: {:.3f}'.format(lr.w_[1]))
print('Intercept: {:.3f}'.format(lr.w_[0]))

# 使用scikit-learn估计回归模型的系数
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Slope: {:.3f}'.format(slr.coef_[0]))
print('Intercept: {:.3f}'.format(slr.intercept_))

lin_reg_plot(X, y, slr)

# 异常值对线性回归模型具有严重的影响
# 随机抽样一致性（Random Sample Consensus, RANSAC）: 即使用数据的一个子集来进行回归模型的拟合
# RANSAC Algo
# 1) 从数据集中随机抽取样本构建内点集合来拟合模型
# 2）使用剩余数据对上一步得到的模型进行测试，并将落在预定公差范围内的样本点增至内点集合中
# 3）使用全部的内点集合数据再次进行模型的拟合
# 4）使用内点集合来估计模型的误差
# 5）如果模型性能达到了用户设定的阈值或者迭代达到了预定次数，则算法终止，否则跳转到第一步

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(base_estimator=LinearRegression(), 
                        max_trials=100,
                        min_samples=50,
                        # residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                        loss='absolute_loss',
                        residual_threshold=5.0,
                        random_state=0)
ransac.fit(X, y)                        

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Slope: {:.3f}'.format(ransac.estimator_.coef_[0]))
print('Intercept: {:.3f}'.format(ransac.estimator_.intercept_))

# 线性回归模型性能的评估
from sklearn.cross_validation import train_test_split
X, y = data, target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# 绘制预测值的残差
plt.scatter(y_train_pred, y_train_pred-y_train, c='blue', marker='o', label='Training Data')
plt.scatter(y_test_pred, y_test_pred -  y_test, c='lightgreen', marker='s', label='Test Data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

# 均方误差衡量模型好坏
from sklearn.metrics import mean_squared_error, r2_score
print('MSE train: {:.3f} vs. test: {:.3f}'.format(
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))

# R-Square衡量模型好坏
print('MSE train: {:.3f} vs. test: {:.3f}'.format(
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))


# 常见正则化线性回归方法
# 1）岭回归(Ridge Regression)  L2罚项
# 2）最小绝对收缩及算子选择(Least Absolute Shrinkage and Selection Operator, LASSO), 适用于稀疏数据训练的模型
# 3）弹性网络(Elastic Net)

# 正则化项不影响截距项w0

from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)    # 如果l1_ratio设置为1，则等同于Lasso回归

# 
# 线性回归模型的曲线化--多项式回归
#

# 增加一个二次多项式项
from sklearn.preprocessing import PolynomialFeatures
X = np.array([258, 270, 294, 
            320, 342, 368,
            396, 446, 480,
            586])[:, np.newaxis]
y = np.array([236, 234.4, 252.8,
            298.6, 314.2, 342.2,
            360.8, 368, 391.2,
            390.8])

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# 拟合一个用于对于的简单线性回归模型
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# 使用经过转换后的特征针对多项式回归拟合一个多元线性回归模型
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', ls='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='best')
plt.show()

# 误差比较
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)

print('Training MSE Linear: {:.3f} vs. Quadratic: {:.3f}'.format(
    mean_squared_error(y, y_lin_pred),
    mean_squared_error(y, y_quad_pred)))

# R-Square衡量模型好坏
print('Training MSE Linear: {:.3f} vs. Quadratic: {:.3f}'.format(
    r2_score(y, y_lin_pred),
    r2_score(y, y_quad_pred)))

#
# 房屋数据集中的非线性关系建模
# 
X = df[['LSTAT']].values
y = target
regr = LinearRegression()

# Creat polynomial features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Linear Fit
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# Quadratic Fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

# Cubic Fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))   # Fitted data
cubic_r2 = r2_score(y, regr.predict(X_cubic))   # Original data

# Plot Results
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear(d=1), $R^2={:.3f}$'.format(linear_r2), color='blue', lw=2, ls=':')
plt.plot(X_fit, y_quad_fit, label='quadratic(d=2), $R^2={:.3f}$'.format(quadratic_r2), color='red', lw=2, ls='-')
plt.plot(X_fit, y_cubic_fit, label='cubic(d=3), $R^2={:.3f}$'.format(cubic_r2), color='green', lw=2, ls='--')
plt.xlabel('Lower status of the population[LSTAT] %')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='best')
plt.show()

# 将多项式映射到线性特征空间，并使用线性回归进行拟合
# Transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# Fit features
X_fit = np.arange(X_log.min()-1, X_log.max() + 1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# Plot results
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear(d=1), $R^2={:.3f}$'.format(linear_r2), color='blue', lw=2, ls=':')
plt.xlabel('log(Lower status of the population(%))')
plt.ylabel("$\sqrt{Price \; in \; \$1000\'s [MEDV]}$")
plt.legend(loc='lower left')
plt.show()

# 
# 使用随机森林处理非线性关系
# 

# 随机森林可以被理解为分段线性函数的集成
# Decision Tree & Information Gain
from sklearn.tree import DecisionTreeRegressor
X = df[['LSTAT']].values
y = target
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_reg_plot(X[sort_idx], y[sort_idx], tree)

# 随机森林对数据集中的异常值不敏感，且无需过多的参数调优
# 所有决策树预测值的平均数作为预测目标变量的值

X, y = data, target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, 
                                criterion='mse',
                                random_state=1,
                                n_jobs=-1)

forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: {:.3f} vs. test: {:.3f}'.format(
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))

print('MSE train: {:.3f} vs. test: {:.3f}'.format(
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))


# 绘制预测值的残差
# 残差没有完全随机分布在中心点附近，这说明模型无法捕获所有的解释信息
plt.scatter(y_train_pred, y_train_pred-y_train, c='black', marker='o', label='Training Data')
plt.scatter(y_test_pred, y_test_pred -  y_test, c='lightgreen', marker='s', label='Test Data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

