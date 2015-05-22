__author__ = 'TerryChen'

import os
import sys
import pandas as pd
import cross_validation as cv

def read_features(read_fn):
    f = open(read_fn, 'r')
    features = []
    for line in f:
        feature = line.strip()
        features.append(feature)
    f.close()
    return features


if __name__ == '__main__':
    model = 'et'
    fn1 = 'general_plus_opt_support_stat.csv'
    fn2 = 'project_eligibility_orgin.csv'
    fn3 = 'exciting_project_rolling_average.csv'
    fn4 = 'resource_cnt.csv'
    fn5 = 'essay_pred_val_3.csv'
    fn6 = 'refined_history_stat.csv'

    input_files = [fn1, fn2, fn3, fn4, fn5, fn6]
    features_fn = 'et1_features.txt'

    features = read_features(features_fn)

    param_grid = {'n_estimators': [2000, 4000],
            'max_features': [2, 5],
            'min_samples_split': [400, 750]}

    parameters = dict()
    parameters['max_leaf_nodes'] = 15

    print 'cv for et1'

    cv.validate(model=model, features=features, parameters=parameters, parameters_grid=param_grid, input_files=input_files)

    sys.exit()