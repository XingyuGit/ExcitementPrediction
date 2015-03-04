from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from pprint import pprint

X, y = make_hastie_10_2(random_state=0)

clf = GradientBoostingClassifier()

param_grid = {'n_estimators': [100, 500], 
            'learning_rate': [0.1, 1.0],
            'max_depth': [1, 3]}

grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print("\nBest score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
pprint(best_parameters)

print "\nGrid score:"
pprint(grid_search.grid_scores_)