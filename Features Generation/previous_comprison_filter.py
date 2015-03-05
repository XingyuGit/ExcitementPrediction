__author__ = 'TerryChen'

#
# Features # 31, 32, 33, 25, 26, 28
#


import sys
import os
import pandas as pd
import numpy as np
sys.path.append('..')
import import_data


def _set_shift(df, key, key_shift):
    # using shift to group by key
    arr = df[key][:len(df) - 1].values
    arr = np.insert(arr, 0, np.nan)
    df[key_shift] = pd.Series(arr, index=df.index)
    df['shift'] = 0
    df.loc[df[key_shift] == df[key], 'shift'] = 1
    return df


def _get_gapdays(df, key):
    # get delta day within group 'key' (sorted by date and key)
    arr = df['date'][:len(df) - 1].values
    arr = np.insert(arr, 0, df['date'][0])
    df['date_prev'] = pd.Series(arr, index=df.index)

    df.loc[df['shift'] != 1, 'date_prev'] = np.nan
    date_object_delta = df['date'] - df['date_prev']
    df['days_since_prev_{}_v2'.format(key)] = date_object_delta / np.timedelta64(1, 'D')
    df.loc[pd.isnull(df['days_since_prev_{}_v2'.format(key)]), 'days_since_prev_{}_v2'.format(key)] = 1

    return df


def _get_prev_price(df):
    # get previous date project cost within group (sorted by date and key)
    arr = df['total_price_including_optional_support'][:len(df) - 1].values
    arr = np.insert(arr, 0, df['total_price_including_optional_support'][0])
    df['price_prev'] = pd.Series(arr, index=df.index)
    df.loc[df['shift'] != 1, 'price_prev'] = np.nan

    return df


def _calculate_total_cost_per_day(df):
    grouped_df = df.sort(['schoolid'], ascending=1).groupby('schoolid')
    # total project cost for each school
    tot_cost = grouped_df['total_price_including_optional_support'].agg(np.sum)

    # calculate #days between the earliest date  and the latest date projectes posted for each school
    days = (grouped_df['date'].agg(np.max) - grouped_df['date'].agg(np.min)) / np.timedelta64(1, 'D') + 1

    # project cost per day for each school
    cost_per_day = tot_cost / days
    
    # convert and merge
    cost_per_day_df = cost_per_day.to_frame(name='tot_cost_per_day_schoolid')
    cost_per_day_df.reset_index(inplace=True)
    merged_df = pd.merge(df, cost_per_day_df, how='left', on='schoolid')
    return merged_df[['projectid', 'tot_cost_per_day_schoolid']]

def _columns_to_write():
    return ['projectid', 'schoolid_gapdays', 'i_price_prev_dif_schoolid_v2',
            'days_since_prev_school_city_v2', 'cost_by_school_city_cummax', 'cost_by_school_zip_cummin']


if __name__ == '__main__':
    # if path is not specified, default is 'Data'
    path = sys.argv[1] if len(sys.argv) > 1 else '../Data'
    projects_df = import_data.get_projects_df(path)
    projects_df = projects_df[
        ['projectid', 'date_posted', 'schoolid', 'school_city', 'school_zip', 'total_price_including_optional_support']]

    projects_df['date'] = pd.to_datetime(projects_df['date_posted'], '%Y-%m-%d')

    df1 = projects_df.sort(['schoolid', 'date'], ascending=[1, 1])
    df1 = _set_shift(df1, 'schoolid', 'school_shift')

    # how many days between the project proposed at date(i) and project at date(i-1) at the same school, default 1
    df1 = _get_gapdays(df1, 'schoolid')

    # project cost (including optional supports) difference between the project at date(i) and project at date(i-1) at the same school
    df1 = _get_prev_price(df1)
    df1['i_price_prev_dif_schoolid_v2'] = df1['total_price_including_optional_support'] - df1['price_prev']
    df1.loc[pd.isnull(df1['i_price_prev_dif_schoolid_v2']), 'i_price_prev_dif_schoolid_v2'] = 1

    df1 = df1[['projectid', 'schoolid_gapdays', 'i_price_prev_dif_schoolid_v2']]

    df2 = projects_df.sort(['school_city', 'date'], ascending=[1, 1])
    df2 = _set_shift(df2, 'school_city', 'city_shift')

    # how many days between the project at date(i) and project at date(i-1) at the same school city
    df2 = _get_gapdays(df2, 'school_city')

    # At date(i), cumulative [from date(0) to date(i-1)] max single project cost (including optional support) at the same school city
    df2 = _get_prev_price(df2)
    df2.loc[pd.isnull(df2['price_prev']), 'price_prev'] = -100
    df2['cost_by_school_city_cummax'] = df2.groupby('school_city')['price_prev'].cummax()
    condition = (df2['cost_by_school_city_cummax'] == -100) | (pd.isnull(df2['cost_by_school_city_cummax']))
    df2.loc[condition, 'cost_by_school_city_cummax'] = 0

    df2 = df2[['projectid', 'days_since_prev_school_city_v2', 'cost_by_school_city_cummax']]

    df3 = projects_df.sort(['school_zip', 'date'], ascending=[1, 1])
    df3 = _set_shift(df3, 'school_zip', 'zip_shift')

    # cost of project at date(i) - cumulative [date(0), ..., date(i-1)] min project cost (including optional support) at the same school zip
    df3 = _get_prev_price(df3)
    df3.loc[pd.isnull(df3['price_prev']), 'price_prev'] = 99999999
    df3['i_price_school_zip_cummin'] = df3.groupby('school_zip')['price_prev'].cummin()
    df3.loc[df3['i_price_school_zip_cummin'] == 99999999, 'i_price_school_zip_cummin'] = np.nan

    df3 = df3[['projectid', 'i_price_school_zip_cummin']]

    # calculate tot cost per day for each school
    df4 = _calculate_total_cost_per_day(projects_df)

    # merge all three data frame with the original one
    projects_df = pd.merge(projects_df, df1, how='left', on='projectid')
    projects_df = pd.merge(projects_df, df2, how='left', on='projectid')
    projects_df = pd.merge(projects_df, df3, how='left', on='projectid')
    projects_df = pd.merge(projects_df, df4, how='left', on='projectid')

    projects_df['cost_by_school_zip_cummin'] = projects_df['i_price_school_zip_cummin'] - projects_df[
        'total_price_including_optional_support']
    projects_df.loc[
        pd.isnull(projects_df['cost_by_school_zip_cummin']), 'cost_by_school_zip_cummin'] = 0

    outputs_df = projects_df[_columns_to_write()]

    # write to csv
    print('writing to csv')
    outputs_df.to_csv(os.path.join('../Features_csv', 'prev_comprisons.csv'), index=False)





