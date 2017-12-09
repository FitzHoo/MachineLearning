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


# Learning Curve
from sklearn.model_selection import learning_curve
pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('clf', LogisticRegression(penalty='l2', random_state=0))
])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train,
                                                       train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, c='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, c='g', ls='--', marker='s', markersize=5, label='test accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='blue')
plt.grid()
plt.xlabel("Number of training samples")
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.ylim([0.8, 1.0])
plt.show()


# Validation Curve  (pipe_lr.get_params().keys() 得到可优化的参数类型)
from sklearn.model_selection import validation_curve
param_range = np.array([0.001, 0.01, 0.1, 1., 10.0, 100.0])
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, param_name='clf__C',
                                             param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, c='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, c='g', ls='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='blue')
plt.grid()
plt.xscale('log')
plt.xlabel("Parameter C")
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.ylim([0.8, 1.0])
plt.show()

# Grid Search(网格搜索）
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([
    ('scl', StandardScaler()),
    ('clf', SVC(random_state=1))
])
param_range = np.logspace(-4, 3, 8)
param_grid = [
    {'clf__C': param_range,
     'clf__kernel': ['linear']},
    {'clf__C': param_range,
     'clf__gamma': param_range,
     'clf__kernel': ['rbf']}
]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
clf.fit(X_test, y_test)
print("Test Accuracy: {:.3f}".format(clf.score(X_test, y_test)))


# 5 x 2 cross validation
gs_ = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
scores = cross_val_score(gs_, X, y, scoring='accuracy', cv=5)   # cv=5 means 5 x 2 cv
print('CV accuracy: {:0.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))


from sklearn.tree import DecisionTreeClassifier
_gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                   param_grid=[
                       {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}
                   ], scoring='accuracy', cv=5)
_scores = cross_val_score(_gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: {:0.3f} +/- {:.3f}'.format(np.mean(_scores), np.std(_scores)))


# Performance Evaluation Indicators  (Precision vs. Recall vs. F1-score)
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[1]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')


# ERR = 1 - ACC
from sklearn.metrics import precision_score, recall_score, f1_score
print("Precision: {:.3f}".format(precision_score(y_true=y_test, y_pred=y_pred)))
print("Recall: {:.3f}".format(recall_score(y_true=y_test, y_pred=y_pred)))
print("F1: {:.3f}".format(f1_score(y_true=y_test, y_pred=y_pred)))


from sklearn.metrics import roc_curve, auc
from scipy import interp
X_train_2 = X_train[:, [4, 14]]  # 选取第4和第14个特征
y_train_2 = y_train[:]
cv = StratifiedKFold(n_splits=3, random_state=1).split(X_train_2, y_train_2)

fig = plt.figure(figsize=(8, 6))
mean_tpr = 0.0   # tpr: true positive rate
mean_fpr = np.linspace(0, 1, 100)   # fpr: false positive rate
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train_2[train], y_train_2[train]).predict_proba(X_train_2[test])
    fpr, tpr, thresholds = roc_curve(y_train_2[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold {} (area = {:.2f})'.format(i+1, roc_auc))
plt.plot([0, 1], [0, 1], ls='--', color=(0.6, 0.6, 0.6), label='random guessing')
mean_tpr /= 3   # 为什么除以3
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = {:.2f}'.format(mean_auc), lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         lw=2, ls=':', c='b', label='perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import accuracy_score, roc_auc_score
pipe_svc = pipe_svc.fit(X_train_2, y_train_2)
y_pred_2 = pipe_svc.predict(X_test[:, [4, 14]])
print("ROC AUC: {:.3f}".format(roc_auc_score(y_true=y_test, y_score=y_pred_2)))
print('Accuracy: {:.3f}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_2)))