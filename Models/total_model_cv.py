__author__ = 'Xingyu Zhou'

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
    model = 'gbm'
    fn1 = 'general_plus_opt_support_stat.csv'
    fn2 = 'project_eligibility_orgin.csv'
    fn3 = 'adjusted_attributes.csv'
    fn4 = 'resource_cnt.csv'
    fn5 = 'essay_pred_val_3.csv'
    fn6 = 'refined_history_stat.csv'
    fn7 = 'cnt_bw_wk_mth_combo.csv'
    fn8 = 'prev_comprisons.csv'
    fn9 = 'exciting_project_rolling_average.csv'

    input_files = [fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9]
    features_fn = 'total_features.txt'

    features = read_features(features_fn)

    param_grid = {
            'n_estimators': [1000],
            'learning_rate': [0.1],
            'subsample': [0.5],
            'min_samples_split': [100],
            'max_leaf_nodes': [7],
    }

    parameters = dict()

    print 'cv for total_model'
    cv.validate(model=model, features=features, parameters=parameters, parameters_grid=param_grid, input_files=input_files)
