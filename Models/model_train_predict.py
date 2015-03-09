__author__ = 'TerryChen'

import os
import sys
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
sys.path.append('..')
import import_data
import time
import random

class train_mode:
    def __init__(self, model, features):
        self._features = features
        #self._features.append(features)
        self._parameters = dict()
        if model != 'gbm' and model != 'et' and model != 'rf':
            raise Exception('invalid model choice. Only "gbm", "et", or "rf" is allowed')
        self.model = model

    def get_model_name(self):
        return self.model

    def set_model_parameters(self, number_trees, learning_rate=0.1, min_sample_split=750, loss='deviance',
                             subsample=0.5, max_leaf_nodes=8, max_features=2):
        """

        :type self: model object
        """
        self._parameters['n_estimators'] = number_trees
        self._parameters['min_samples_split'] = min_sample_split
        self._parameters['max_leaf_nodes'] = max_leaf_nodes
        self._parameters['verbose'] = 1

        if self.model == 'gbm':
            self._parameters['learning_rate'] = learning_rate
            self._parameters['loss'] = loss
            self._parameters['subsample'] = subsample
        else:
            self._parameters['max_features'] = max_features


    def train_and_predict(self, input_files, output_fn, data_path='../Data'):
        """

        :param input_files:  a list of features files
        :param output_fn: filename of output file (including .csv)
        :param data_path: path of origin data
        """
        random.seed()

        start_time =  time.time()
        # #  obtain complete data set for training
        outcomes_df = import_data.get_outcomes_df(data_path)
        projects_df = import_data.get_projects_df(data_path)
        df = pd.merge(projects_df, outcomes_df, how='left', on='projectid')[['projectid', 'group', 'y']]
        df.columns = ['projectid', 'dataset', 'outcome_y']
        for x in range(len(input_files)):
            input_df = pd.read_csv(os.path.join('../Features_csv', input_files[x]))
            df = pd.merge(df, input_df, how='left', on='projectid', suffixes=('', '_x'))

        # split into train data and test data

        x_train_df = df[self._features][df['dataset'] != 'test']
        y_train_df = df['outcome_y'][df['dataset'] != 'test']
        x_test_df = df[self._features][df['dataset'] == 'test']

        df2 = df[self._features]
        for var in self._features:
            d = df2[var][pd.isnull(df2[var])]
            if d.size > 0:
                print(var)
        print df2.size

        # convert to 2D array
        x_train_matrix = x_train_df.as_matrix()
        y_train_matrix = y_train_df.as_matrix()
        x_test_matrix = x_test_df.as_matrix()

        #   train model
        print('building model')
        if self.model == 'gbm':
            print 'builidng GBM'
            clf = GradientBoostingClassifier(**self._parameters)
        elif self.model == 'et':
            print 'building extra trees'
            clf = ExtraTreesClassifier(**self._parameters)
        else:
            print 'building random forest'
            clf = RandomForestClassifier(**self._parameters)
        print('training model')
        clf.fit(x_train_matrix, y_train_matrix)

        end_time = time.time()
        print('build and train model: %0.3f seconds' % (end_time - start_time))

        #  predict and convert to dataframe for writing
        print('predict values')
        y_all_matrix = clf.predict_proba(df[self._features].as_matrix())
        # y_test_matrix = clf.predict_proba(x_test_matrix)
        # 2nd column (probability of 1)
        y_all_matrix = y_all_matrix[:,1] 
        # y_test_matrix = y_test_matrix[:,1]

        # y_all_df = pd.DataFrame({'{}_all_y'.format(output_fn[:-4]): y_all_matrix})
        #y_all_df = pd.DataFrame(y_all_matrix, columns='{}_all_y'.format(output_fn[:-4]))
        # y_test_df = pd.DataFrame({'{}_y'.format(output_fn[:-4]): y_test_matrix})
        #y_test_df = pd.DataFrame(y_test_matrix, columns='{}_y'.format(output_fn[:-4]))

        df['{}_y'.format(output_fn[:-4])] = y_all_matrix

        output_all_df = df[['projectid', '{}_y'.format(output_fn[:-4])]]
        output_y_df = df[['projectid', '{}_y'.format(output_fn[:-4])]][df['dataset'] == 'test']
        output_y_df.columns = ['projectid', 'is_exciting']
        # write to file
        print('writing file: ' + output_fn)
        output_y_df.to_csv(os.path.join('../Prediction', output_fn), index=False)
        output_all_df.to_csv(os.path.join('../Prediction', output_fn[:-4] + '_all.csv'), index=False)


