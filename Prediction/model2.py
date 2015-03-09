__author__ = 'TerryChen'

import os
import sys
sys.path.append('..')
import import_data as im
import pandas as pd

if __name__ == '__main__':

    projects_df = im.get_projects_df('../Data')
    projects_df = projects_df[projects_df['group'] == 'test']

    gbm = pd.read_csv(os.path.join('../Prediction', 'gbm5_predict.csv'))
    et = pd.read_csv(os.path.join('../Prediction', 'et2_predict.csv'))
    rf = pd.read_csv(os.path.join('../Prediction', 'rf1_predict.csv'))

    projects_df['is_exciting'] = 0.45 * gbm['is_exciting'] + 0.45 * et['is_exciting'] + 0.1 * rf['is_exciting']

    projects_df = projects_df[['projectid', 'is_exciting']]

    projects_df.to_csv(os.path.join('../Prediction', 'model2.csv'), index=False)