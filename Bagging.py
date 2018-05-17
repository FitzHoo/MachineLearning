import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
data = load_wine()

wine_df = pd.DataFrame(data.data, columns=data.feature_names)
wine_df['class_label'] = data.target
wine_df = wine_df[wine_df['class_label'] != 1]
X = wine_df[['alcohol', 'hue']].values
y = wine_df['class_label'].values

# 将类标编码为二进制形式
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0,
                        bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print("Decision tree train/test accuracies {:.3f} / {:.3f}".format(tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print("Bag train/test accuracies {:.3f} / {:.3f}".format(bag_train, bag_test))


# 画出的图像有点问题
def plot_contourf_zone(classifiers={'Decision Tree': tree, 'Bagging': bag}):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() - 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))
    for idx, clf, tt in zip([0, 1], classifiers.values(), classifiers.keys()):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train == 0, 0],
                           X_train[y_train == 0, 1],
                           c='blue', marker='^')
        axarr[idx].scatter(X_train[y_train == 1, 0],
                           X_train[y_train == 1, 1],
                           c='red', marker='o')
        axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Hue', fontsize=12)
    plt.text(10.2, -1.2, s="Alcohol", ha='center', va='center', fontsize=12)
    return plt.show()


# AdaBoost
'''
1. 以等值方式为权重向量w赋值，其中权重之和为1
2. 在m轮boosting操作中，对第j轮做如下操作：
    2.1 训练一个加权的弱学习器：Cj = train(X, y, w)
    2.2 预测样本类标：y_head = predict(Cj, X)
    2.3 计算加权错误率：epsilon = w * (y - y_head)
    2.4 计算相关系数： alpha_j = 0.5 * (1 - epsilon) / epsilon
    2.5 更新权重：w = w * exp(-alpha_j * y * y_head)
    2.6 归一化权重 
    2.7 进行新的预测，重复步骤2直到程序结束
    
'''
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print("Decision tree train/test accuracies {:.3f} / {:.3f}".format(tree_train, tree_test))
print("AdaBoost train/test accuracies {:.3f} / {:.3f}".format(ada_train, ada_test))

plot_contourf_zone(classifiers={'Decision Tree': tree, 'AdaBoost': ada})



