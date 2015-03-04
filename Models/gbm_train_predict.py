__author__ = 'TerryChen'

import os
import sys
import pandas as pd
sys.path.append('..')
from sklearn.ensemble import GradientBoostingClassifier

import import_data

_features = ['projectid']
_parameters = dict()
_input_files = []
_output_fn = ''
_data_path = '../Data'

def set_training_features(features):
    for x in range(len(features)):
        _features.append(x)

def set_model_parameters(number_trees, learning_rate=0.1, min_sample_split=750, loss='deviance', subsample=0.5, max_leaf_nodes=100):
    _parameters['n_estimators'] = number_trees
    _parameters['shrinkage'] = learning_rate
    _parameters['min_split'] = min_sample_split
    _parameters['loss'] = loss
    _parameters['subsample'] = subsample
    _parameters['max_leaf_nodes'] = max_leaf_nodes

def set_input_output_files(input_files, output_fn, data_path):
    _input_files = input_files
    _output_fn = output_fn

def train_and_predict():
    outcomes_df = import_data.get_outcomes_df(_data_path)
    projects_df = import_data.get_projects_df(_data_path)
    df = pd.merge(projects_df, outcomes_df, how='left', on='projectid')[['projectid', 'group', 'y']]
    for x in range(len(_input_files)):
        input_df = pd.read_csv(os.path.join('../Features_csv'), _input_files[x])
        df = pd.merge(df, input_df, how='left', on='projectid')

    x_train_df = df[_features][(df['group'] == 'valid') | (df['group'] == 'train')]
    y_train_df = df['y'][(df['group'] == 'valid') | (df['group'] == 'train')]
    x_test_df = df[_features][df['group'] == 'test']

    x_train_matrix = x_train_df.as_matrix()
    y_train_matrix = y_train_df.as_matrix()
    x_test_matrix = x_test_df.as_matrix()

    clf = GradientBoostingClassifier(n_estimators=_parameters['n_estimators'], loss=_parameters['loss'], min_samples_split=_parameters['min_split'], subsample=_parameters['subsample'], max_leaf_nodes=_parameters['max_leaf_nodes'])
    clf.fit(x_train_matrix, y_train_matrix)

    y_test_matrix = clf.predict(x_test_matrix)
    y_test_df = pd.DataFrame(y_test_matrix, columns='y')

    # write to file
    print('writing file: ' + _output_fn)
    y_test_df.to_csv(os.path.join('../Prediction'), _output_fn)

    sys.exit()