__author__ = 'TerryChen'

import os
import sys
sys.path.append('..')
import import_data
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from pprint import pprint


def validate(model, features, parameters, parameters_grid, input_files):
    data_path = '../Data'
    outcomes_df = import_data.get_outcomes_df(data_path)
    projects_df = import_data.get_projects_df(data_path)
    df = pd.merge(projects_df, outcomes_df, how='left', on='projectid')[['projectid', 'group', 'y']]
    for x in range(len(input_files)):
        input_df = pd.read_csv(os.path.join('../Features_csv'), input_files[x])
        df = pd.merge(df, input_df, how='left', on='projectid')

    x_train_df = df[features][(df['group'] == 'valid') | (df['group'] == 'train')]
    y_train_df = df['y'][(df['group'] == 'valid') | (df['group'] == 'train')]

    if model is 'gbm':
        clf = GradientBoostingClassifier(**parameters)
    elif model is 'et':
        clf = ExtraTreesClassifier(**parameters)
    else:
        clf = RandomForestClassifier(**parameters)

    grid_search = GridSearchCV(clf, parameters_grid, n_jobs=-1, verbose=1)
    grid_search.fit(x_train_df.as_matrix(), y_train_df.as_matrix())

    print("\nBest score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    pprint(best_parameters)

    print "\nGrid score:"
    pprint(grid_search.grid_scores_)
