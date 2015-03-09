__author__ = 'TerryChen'

import os
import sys
sys.path.append('..')
import import_data as im
import pandas as pd
import numpy as np

if __name__ == '__main__':

    projects_df = im.get_projects_df('../Data')
    projects_df = projects_df[projects_df['group'] == 'test']
    projects_df = projects_df[['projectid', 'date_posted']]
    ref_date = pd.to_datetime('2014-05-12', '%Y-%m-%d')
    projects_df['date'] = pd.to_datetime(projects_df['date_posted'], '%Y-%m-%d')

    projects_df['date_diff'] = (ref_date - projects_df['date']) / np.timedelta64(1, 'D')

    lower = 0.5
    slope = ( 1 - lower) / 131
    projects_df['discount'] = projects_df['date_diff'] * slope + lower

    gbm1 = pd.read_csv(os.path.join('../Prediction', 'gbm1_predict.csv'))
    gbm2 = pd.read_csv(os.path.join('../Prediction', 'gbm2_predict.csv'))
    gbm3 = pd.read_csv(os.path.join('../Prediction', 'gbm3_predict.csv'))
    gbm4 = pd.read_csv(os.path.join('../Prediction', 'gbm4_predict.csv'))
    et1 = pd.read_csv(os.path.join('../Prediction', 'et1_predict.csv'))

    projects_df['pred1'] = 0.1 * gbm1['is_exciting'] + 0.1 * gbm2['is_exciting'] + 0.45 * gbm3['is_exciting'] + 0.1 * gbm4['is_exciting'] + 0.25 * et1['is_exciting']
    projects_df['is_exiciting'] = projects_df['pred1'] * projects_df['discount']


    outcome_df_dis = projects_df[['projectid', 'is_exciting']]
    outcome_df_dis.to_csv(os.path.join('../Prediction', 'model1_w_discount.csv'), index=False)

    outcome_df = projects_df[['projectid', 'pred1']]
    outcome_df.columns = ['projectid', 'is_exciting']
    outcome_df.to_csv(os.path.join('../Prediction', 'model1.csv'), index=False)
