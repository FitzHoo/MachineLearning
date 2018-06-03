'''
假定每个分类器都是独立的，且出错率之间是不相关的，则集成分类器的出错概率可以简单地表示为二项分布的概率密度函数（伯努利分布）
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb
import math


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.0))
    probs = [comb(n_classifier, k) * error ** k * (1-error) ** (n_classifier - k)
             for k in range(k_start, n_classifier+1)]
    return sum(probs)


error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
plt.plot(error_range, ens_errors, label='Ensemble Error', lw=2)
plt.plot(error_range, error_range, label='Base Error', ls='--', lw=2)
plt.xlabel('Base Error')
plt.ylabel('Base/Ensemble Error')
plt.legend(loc='best')
plt.grid()
plt.show()

# Majority Voting
ex = np.array([
    [0.9, 0.1],
    [0.8, 0.2],
    [0.4, 0.6]
])
weights = [0.2, 0.2, 0.6]
p = np.average(ex, axis=0, weights=weights)
indice = np.argmax(p)

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import _name_estimators


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):   # 多重继承
    '''
    A majority vote ensemble classifier

    Paramters:
    ==========
    classifiers:
        array-like, shape=[n_classifier]
        Different classifiers for the ensemble

    vote:
        str, {'classlabel', 'probability'}
        Default: 'classlabel'
        If 'classlabel' the prediction is based on the argmax of class labels. Else if 'probability', the
        argmax of the sum of probability is used to predict the class label(recommended for
        calibrated classifiers).

    weights:
        array-like, shape=[n_classifiers]
        Optional, default: None
        If a list of 'int' or 'float' values are provided, the classifiers are weighted by importance;
        Uses uniform weights if 'weights=None'

    '''

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        # Use LabelEncoder to ensure class labels start with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)

        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_probas = np.average(probas, weights=self.weights, axis=0)
        return avg_probas

    def get_params(self, deep=True):   # Overrides method in BaseEstimator
        '''
        Get classifier parameters names for GridSearch
        '''
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)    # 继承
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['{}__{}'.format(name, key)] = value
            # for name, step in six.iteritems(self.named_classifiers):
            #     for key, value in six.iteritems(step.get_params(deep=True)):
            #         out['{}__{}'.format(name, key)] = value
            return out


from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# iris = datasets.load_iris()
# X, y = iris.data[50:, [1, 2]], iris.target[50:]
# le = LabelEncoder()
# y = le.fit_transform(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
#
# # 使用三种不同的分类器来训练数据：Logistic Regression, KNN, Decision Tree
#
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
#
# clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
# clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
# clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
# pipe1 = Pipeline([
#     ['sc', StandardScaler()],
#     ['clf', clf1]
# ])
#
# # pipe2 = Pipeline([
# #     ['clf', clf2]
# # ])
#
# pipe3 = Pipeline([
#     ['sc', StandardScaler],
#     ['clf', clf3]
# ])
# clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
# print('10-fold cross validation: \n')
# for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
#     scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
#     print('ROC AUC: {:.2f} (+/-) {:.2f} {}'.format(scores.mean(), scores.std(), label))


# 逻辑斯蒂回归于K近邻算法对数据缩放不敏感
# mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
# clf_labels += ['Majority Voting']
# all_clf = [pipe1, clf2, pipe3, mv_clf]
# for clf, label in zip(all_clf, clf_labels):
#     scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
#     print('ROC AUC: {:.2f} (+/-) {:.2f} {}'.format(scores.mean(), scores.std(), label))


# 评估与调优集成分类器
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# colors = ['black', 'orange', 'blue', 'green']
# linestyles = [':', '--', '-.', '-']
# for clf, label, color, ls in zip(all_clf, clf_labels, colors, linestyles):
#     # assume the label of the positive class is 1
#     y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
#     fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
#     roc_auc = auc(x=fpr, y=tpr)
#     plt.plot(fpr, tpr, color=color, ls=ls, label='{} (auc={:.2f})'.format(label, roc_auc))
#
# plt.legend(loc='best')
# plt.plot([0, 1], [0, 1], ls='--', color='gray', lw=2)
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])
# plt.grid()
# plt.xlabel('false positive rate')
# plt.ylabel('true positive rate')
# plt.show()

# Some bugs to be fixed in the future, and right now just jump over this chapter

#
# Bagging
#
# 通过Bootstrap样本构建集成分类器
# Bagging suanfa 算法是降低模型方差的一种有效方法，然而Bagging在降低模型偏差方面的作用不大
wine_data = datasets.load_wine()
X = wine_data.data[:, [0, 10]]
y = wine_data.target

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)

from sklearn.metrics import accuracy_score
tree_ = tree.fit(X_train, y_train)
y_train_pred = tree_.predict(X_train)
y_test_pred = tree_.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision Tree train/test accuracies {:.3f} vs {:.3f}'.format(tree_train, tree_test))

bag_ = bag.fit(X_train, y_train)
y_train_pred = bag_.predict(X_train)
y_test_pred = bag_.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies {:.3f} vs {:.3f}'.format(bag_train, bag_test))

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision Tree', 'Bagging']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
                       c='blue', marker='o')
    axarr[idx].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
                       c='red', marker='o')
    axarr[idx].scatter(X_train[y_train==2, 0], X_train[y_train==2, 1],
                       c='green', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(5, -1, s='Hue', ha='center', va='center', fontsize=12)
plt.show()


#
# Adaboost
#
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=0)
_tree = tree.fit(X_train, y_train)
y_train_pred = _tree.predict(X_train)
y_test_pred = _tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision Tree train/test accuracies {:.3f} vs {:.3f}'.format(tree_train, tree_test))

_ada = ada.fit(X_train, y_train)
y_train_pred = _ada.predict(X_train)
y_test_pred = _ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies {:.3f} vs {:.3f}'.format(ada_train, ada_test))


f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
                       c='blue', marker='o')
    axarr[idx].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
                       c='red', marker='o')
    axarr[idx].scatter(X_train[y_train==2, 0], X_train[y_train==2, 1],
                       c='green', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(5, -1, s='Hue', ha='center', va='center', fontsize=12)
plt.show()