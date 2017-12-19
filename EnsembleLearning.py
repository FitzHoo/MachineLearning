'''
假定每个分类器都是独立的，且出错率之间是不相关的，则集成分类器的出错概率可以简单地表示为二项分布的概率密度函数（伯努利分布）
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb
import math


def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * error ** k * (1-error) ** (n_classifier - k) for k in range(k_start, n_classifier+1)]
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
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six     # 与Python2.7兼容
from sklearn.pipeline import _name_estimators
import operator


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
        self.lablenc_.fit(X, y)
        self.classes = self.labelnc_.classes_
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
            return super(MajorityVoteClassifier, self).get_params(deep=False)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# 使用三种不同的分类器来训练数据：Logistic Regression, KNN, Decision Tree

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([
    ['sc', StandardScaler()],
    ['clf', clf1]
])
pipe3 = Pipeline([
    ['sc', StandardScaler],
    ['clf', clf3]
])
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-fold cross validation: \n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print('ROC AUC: {:.2f} (+/-) {:.2f} {}'.format(scores.mean(), scores.std(), label))
