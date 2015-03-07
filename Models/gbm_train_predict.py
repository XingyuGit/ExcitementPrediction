__author__ = 'TerryChen'

import os
import sys
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
sys.path.append('..')
import import_data


class gbm:
    def __init__(self, features):
        self._features = ['projectid']
        self._features.append(features)
        self._parameters = dict()


    def set_model_parameters(self, number_trees, learning_rate=0.1, min_sample_split=750, loss='deviance',
                             subsample=0.5, max_leaf_nodes=8):
        """

        :type self: gbm object
        """
        self._parameters['n_estimators'] = number_trees
        self._parameters['learning_rate'] = learning_rate
        self._parameters['min_samples_split'] = min_sample_split
        self._parameters['loss'] = loss
        self._parameters['subsample'] = subsample
        self._parameters['max_leaf_nodes'] = max_leaf_nodes


    def train_and_predict(self, input_files, output_fn, data_path='../Data'):
        """

        :param input_files:  a list of features files
        :param output_fn: filename of output file (including .csv)
        :param data_path: path of origin data
        """

        # #  obtain complete data set for training
        outcomes_df = import_data.get_outcomes_df(data_path)
        projects_df = import_data.get_projects_df(data_path)
        df = pd.merge(projects_df, outcomes_df, how='left', on='projectid')[['projectid', 'group', 'y']]
        for x in range(len(input_files)):
            input_df = pd.read_csv(os.path.join('../Features_csv'), input_files[x])
            df = pd.merge(df, input_df, how='left', on='projectid')

        # split into train data and test data
        x_train_df = df[self._features][(df['group'] == 'valid') | (df['group'] == 'train')]
        y_train_df = df['y'][(df['group'] == 'valid') | (df['group'] == 'train')]
        x_test_df = df[self._features][df['group'] == 'test']

        # convert to 2D array
        x_train_matrix = x_train_df.as_matrix()
        y_train_matrix = y_train_df.as_matrix()
        x_test_matrix = x_test_df.as_matrix()

        #   train model
        clf = GradientBoostingClassifier(**self._parameters)
        clf.fit(x_train_matrix, y_train_matrix)

        #   predict and convert to dataframe for writing
        y_test_matrix = clf.predict(x_test_matrix)
        y_test_df = pd.DataFrame(y_test_matrix, columns='y')

        # write to file
        print('writing file: ' + output_fn)
        y_test_df.to_csv(os.path.join('../Prediction'), output_fn)
