from time import time
from scipy.stats import randint

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_digits


def search_cv():
    digits = load_digits(n_class=10)
    X, y = digits.data, digits.target

    rfc = RandomForestClassifier(n_estimators=20)

    # Randomized Search
    param_dist = {'max_depth': [3, 5],
                  'max_features': randint(1, 11),
                  'min_samples_split': randint(2, 11),
                  'criterion': ['gini', 'entropy']}

    n_iter_search = 20
    random_search = RandomizedSearchCV(rfc, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)

    start = time()
    random_search.fit(X, y)
    print('RandomizedSearchCV {:.2f} seconds for {:d} candidates parameter settings'.format(time() - start,
                                                                                            n_iter_search))
    print(random_search.best_params_)
    print(random_search.best_score_)

    # Grid Search
    param_grid = {'max_depth': [3, 5],
                  'max_features': [1, 3, 10],
                  'min_samples_split': [2, 3, 10],
                  'criterion': ['gini', 'entropy']}

    grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)

    start = time()
    grid_search.fit(X, y)
    print('GridSearchCV {:.2f} seconds for {:d} candidates parameter settings'.format(
        time() - start, len(grid_search.cv_results_['params'])))
    print(grid_search.best_params_)
    print(grid_search.best_score_)


if __name__ == '__main__':
    search_cv()
