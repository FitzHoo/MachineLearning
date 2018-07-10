import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

clf = SVC(kernel='linear', C=1)
clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)
print(accuracy_score(test_y, pred_y))

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), SVC(C=1.0, kernel='linear'))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error {:.2f}'.format(-1 * scores.mean()))

from xgboost import XGBClassifier
new_pipeline = Pipeline([('imputer', Imputer()), ('xgbrg', XGBClassifier())])

from sklearn.model_selection import GridSearchCV
# 寻优的参数选择
param_grid = {
    'xgbrg__n_estimators': [10, 50, 100, 500],
    'xgbrg__learning_rate': [0.1, 0.5, 1],
}

# dict, optional
# Parameters to pass to fit method
fit_params = {
    'xgbrg__eval_set': [(test_X, test_y)],
    'xgbrg__early_stopping_rounds': 10,
    'xgbrg__verbose': False
}

# 5-fold cross validation by passing the argument cv=5
searchCV = GridSearchCV(new_pipeline, cv=5, param_grid=param_grid, fit_params=fit_params)
searchCV.fit(train_X, train_y)

print(searchCV.best_params_)
print(searchCV.cv_results_['mean_train_score'])
print(searchCV.cv_results_['mean_test_score'])

