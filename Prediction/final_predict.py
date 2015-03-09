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

    model1 = pd.read_csv(os.path.join('../Prediction', 'model1.csv'))
    model2 = pd.read_csv(os.path.join('../Prediction', 'model2.csv'))

    projects_df['is_exciting'] = (0.5 * model1['is_exciting'] + 0.5 * model2['is_exciting']) * projects_df['discount']

    projects_df = projects_df[['projectid', 'is_exciting']]

    projects_df.to_csv(os.path.join('../Prediction', 'final_result.csv'))