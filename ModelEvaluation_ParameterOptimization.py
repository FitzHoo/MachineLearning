import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()
X = breast_cancer_data.data
y = breast_cancer_data.target

# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Pipeline Workflow
from sklearn.preprocessing import StandardScaler   # 数据标准化
from sklearn.decomposition import PCA              # 数据特征提取
from sklearn.linear_model import LogisticRegression    # 数据训练
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print("Test Accuracy: {:.3f}".format(pipe_lr.score(X_test, y_test)))

# holdout cross validation
# k-fold cross validation   vs.  stratified k-fold cross validation
# leave-one-out, LOO  # 适用小样本
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=1)
scores = []
for k, (train, test) in enumerate(kfold.split(X, y)):
    pipe_lr.fit(X[train], y[train])
    score = pipe_lr.score(X[test], y[test])
    scores.append(score)
    print('Fold; {}, Class dist.: {}, Acc: {:.3f}'.format(k+1, np.bincount(y[train]), score))
print("CV accuracy: {:.3f} +/- {:.3f}".format(np.mean(scores), np.std(scores)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr, X=X, y=y, cv=10, n_jobs=1)
print("CV accuracy: {:.3f} +/- {:.3f}".format(np.mean(scores), np.std(scores)))


# Learning Curve & Validation Curve


