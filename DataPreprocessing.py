import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Construct a man-made data
df = pd.DataFrame(np.array([
    [1., 2., 3., 4.],
    [5., 6., np.nan, 8.],
    [10., 11., 12., np.nan]
]), columns=list('ABCD'))


# Imputation 插值
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data


df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
], columns=['color', 'size', 'price', 'classlabel'])

# Convert ordinal feature to int number
size_mapping = {
    'XL': 3,
    'L': 2,
    "M": 1
}

inv_size_mapping = {v: k for k, v in size_mapping.items()}

df['size'] = df['size'].map(size_mapping)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
inv_class_mapping = {v: k for k, v in class_mapping.items()}

df['classlabel'] = df['classlabel'].map(class_mapping)

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

color_re = LabelEncoder()
df.ix[:, 0] = color_re.fit_transform(df.ix[:, 0].values)


# One-hot encoding  == pd.get_dummpy()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0], sparse=True)
ohe.fit_transform(df.values).toarray()

from sklearn.datasets import load_wine
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
wine_df['target'] = wine_data['target']

from sklearn.model_selection import train_test_split
X, y = wine_df.iloc[:, :-1].values, wine_df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Normalization
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

# Standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print("Training Accuracy:", lr.score(X_train_std, y_train))
print("Test Accuracy:", lr.score(X_test_std, y_test))


# Plot Regularization Weights Variation
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta',
          'yellow', 'black', 'pink', 'lightgreen',
          'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)
weights = np.array(weights)
for col, color in zip(range(weights.shape[1]), colors):
    ax.plot(params, weights[:, col], label=wine_df.columns[col], color=color)
plt.axhline(0, color='black', ls='--', lw=3)
plt.xlim([10 ** (-5), 10 ** 5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()


# Sequential Backward Selection
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score

class SBS(object):
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25,
                 random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1]  # feature number
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)

            dim -= 1
            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# Test the top five significant features
k5 = list(sbs.subsets_[8])   # 8 = 13 - 5
print(wine_df.columns[k5])

knn.fit(X_train_std, y_train)
print("Training Accuracy:", knn.score(X_train_std, y_train))
print('Test Accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k5], y_train)
print("Training Accuracy:", knn.score(X_train_std[:, k5], y_train))
print('Test Accuracy:', knn.score(X_test_std[:, k5], y_test))


# judge the feature importance by random forest
from sklearn.ensemble import RandomForestClassifier
feat_labels = wine_df.columns[:-1]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indicies = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print('%2d) %-*s %f' %(f+1, 30, feat_labels[f], importances[indicies[f]]))

plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indicies], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.tight_layout()
plt.show()


