__author__ = 'TerryChen'

import os
import sys
sys.path.append('..')
import import_data as im
import pandas as pd
import numpy as np

if __name__ == '__main__':
    flag = False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'factor':
            flag = True

    projects_df = im.get_projects_df('../Data')
    outcomes_df = im.get_outcomes_df('../Data')
    donations_df = im.get_donations_df('../Data')

    if flag:

        donations_df = pd.merge(donations_df, projects_df, how='left', on='projectid')
        donations_df['d_date'] = donations_df['donation_timestamp'].str[:10]
        donations_df['date'] = pd.to_datetime(donations_df['d_date'], '%Y-%m-%d')
        max_date_df = donations_df.groupby('projectid')['date'].agg(np.max).to_frame(name='latest_date')
        max_date_df.reset_index(inplace=True)
        df = pd.merge(projects_df, max_date_df, how='left', on='projectid')
        df['date'] = pd.to_datetime(df['date_posted'], '%Y-%m-%d')
        df['days_to_fully_funding'] = (df['latest_date'] - df['date']) / np.timedelta64(1, 'D')

        df = pd.merge(df, outcomes_df, how='left', on='projectid')
        df = df[(df['date_posted'] >= '2013-10-01') & (df['date_posted'] < '2014-01-01') & (df['fully_funded'] == 1)]
        grouped = df.groupby('days_to_fully_funding')
        df_cnt = grouped.size().to_frame(name='proj_cnt')
        df_cnt.reset_index(inplace=True)
        df_is_cnt = grouped['y'].agg(np.sum).to_frame(name='proj_is_cnt')
        df_is_cnt.reset_index(inplace=True)
        df = pd.merge(df, df_cnt, how='left', on='days_to_fully_funding')
        df = pd.merge(df, df_is_cnt, how='left', on='days_to_fully_funding')

        df = df[(df['days_to_fully_funding'] >= 0) & (df['days_to_fully_funding'] <= 120) & (pd.notnull(df['days_to_fully_funding']))]
        df = df.sort('date', ascending=1)
        df.loc[pd.isnull(df['proj_cnt']), 'proj_cnt'] = 0
        df.loc[pd.isnull(df['proj_is_cnt']), 'proj_is_cnt'] = 0
        df['proj_cumsum'] = np.cumsum(df['proj_cnt'])
        df['proj_is_cumsum'] = np.cumsum(df['proj_is_cnt'])
        df['proj_is_pct'] = df['proj_is_cumsum'] / df['proj_cumsum']
        df['factor'] = df['proj_is_pct'] / np.max(df['proj_is_pct'])
        df = df[['days_to_fully_funding', 'factor']]
        df.columns = ['date_diff', 'factor']

        projects_df['date'] = pd.to_datetime(projects_df['date_posted'], '%Y-%m-%d')
        max_date = np.max(projects_df['date'])
        projects_df['date_diff'] = (max_date - projects_df['date']) / np.timedelta64(1, 'D')
        projects_df = pd.merge(projects_df, df, how='left', on='date_diff')

    projects_df = projects_df[projects_df['group'] == 'test']

    gbm = pd.read_csv(os.path.join('../Prediction', 'gbm5_predict.csv'))
    et = pd.read_csv(os.path.join('../Prediction', 'et2_predict.csv'))
    rf = pd.read_csv(os.path.join('../Prediction', 'rf1_predict.csv'))

    if flag:
        projects_df['pred'] = (1 * gbm['is_exciting'] + 1 * et['is_exciting'] + 0.2 * rf['is_exciting'])/2.2
        projects_df['is_exciting'] = projects_df['pred']
        for i in range(projects_df['projectid'].size):
            if pd.notnull(projects_df.loc[i, 'factor']):
                projects_df.loc[i, 'is_exciting'] = projects_df.loc[i, 'factor'] * projects_df.loc[i, 'pred']

    else:
        projects_df['is_exciting'] = 0.45 * gbm['is_exciting'] + 0.45 * et['is_exciting'] + 0.1 * rf['is_exciting']


    projects_df = projects_df[['projectid', 'is_exciting']]

    projects_df.to_csv(os.path.join('../Prediction', 'model2.csv'), index=False)