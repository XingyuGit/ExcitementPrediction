__author__ = 'TerryChen'

#
#   Features # 20 - 24, 7
#

import sys
import os
import pandas as pd
import datetime
import numpy as np
sys.path.append('..')
import import_data

def _timedelta(d):
    return datetime.timedelta(days=d)


def _cnt_wk_bwk_mth_combination(df, key):
    print('Compute count for {} and date pair for a week, two weeks, and a month'.format(key))

    # set attribute name keyword
    key_abbr = key
    if key == 'school_zip':
        key_abbr = 'zip'
    elif key == 'school_city':
        key_abbr = 'city'

    # sum of project in the same date and same city/zip/school
    tmp = df.groupby(['date', key]).size().to_frame(name='cnt_day_{}'.format(key))
    tmp.reset_index(inplace=True)
    df = pd.merge(df, tmp, how= 'left', on=[key, 'date'])

    # calculate date +1, ..., +6, -1, ..., -6
    for i in range(6):
        tmp.columns = ['date+{}'.format(i + 1), key, 'cnt_day_{}+{}'.format(key_abbr, i + 1)]
        df = pd.merge(df, tmp, how= 'left', on= [key, 'date+{}'.format(i + 1)])
        filter = pd.isnull(df['cnt_day_{}+{}'.format(key_abbr, i + 1)])
        df.loc[filter, 'cnt_day_{}+{}'.format(key_abbr, i + 1)] = 0

        tmp.columns = ['date-{}'.format(i + 1), key, 'cnt_day_{}-{}'.format(key_abbr, i + 1)]
        df = pd.merge(df, tmp, how= 'left', on= [key, 'date-{}'.format(i + 1)])
        filter = pd.isnull(['cnt_day_{}-{}'.format(key_abbr, i + 1)])
        df.loc[filter, 'cnt_day_{}-{}'.format(key_abbr, i + 1)] = 0

    # sum of project in the same month and same city
    tmp = df.groupby(['yearmonth', key]).size().to_frame(name='cnt_monthly_by_{}'.format(key_abbr))
    tmp.reset_index(inplace=True)
    df = pd.merge(df, tmp, how='left', on=[key, 'yearmonth'])

    # sum for total counts
    cnt = df['cnt_day_' + key]
    for i in range(3):
        cnt += df['cnt_day_{}+{}'.format(key_abbr, i + 1)]
        cnt += df['cnt_day_{}-{}'.format(key_abbr, i + 1)]
    df['cnt_weekly_by_' + key_abbr] = cnt

    cnt = df['cnt_day_' + key]
    for i in range(6):
        cnt += df['cnt_day_{}+{}'.format(key_abbr, i + 1)]
        cnt += df['cnt_day_{}-{}'.format(key_abbr, i + 1)]
    df['cnt_biweekly_by_' + key_abbr] = cnt

    return df

def _columns_to_write():
    return ['projectid', 'ave_proj_cost_school_city', 'cnt_weekly_by_schoolid', 'cnt_biweekly_by_schoolid', 'cnt_monthly_by_schoolid', 'cnt_weekly_by_zip', 'cnt_biweekly_by_zip', 'cnt_monthly_by_zip', 'cnt_weekly_by_city', 'cnt_biweekly_by_city', 'cnt_monthly_by_city']


if __name__ == '__main__':
    # if path is not specified, default is 'Data'
    path = sys.argv[1] if len(sys.argv) > 1 else '../Data'
    projects_df = import_data.get_projects_df(path)
    projects_df = projects_df[['projectid', 'schoolid', 'date_posted', 'school_city', 'school_zip', 'total_price_excluding_optional_support']]

    # delta of date
    date_delta = [_timedelta(1), _timedelta(2), _timedelta(3), _timedelta(4), _timedelta(5), _timedelta(6)]
    projects_df['date'] = pd.to_datetime(projects_df['date_posted'], '%Y-%m-%d')
    projects_df['yearmonth'] = pd.to_datetime(projects_df['date_posted'].str[:7], '%Y-%m')

    # introduce columns of dates from current_date + 6 to current_date - 6
    for i in range(6):
        projects_df['date+' + str(i + 1)] = projects_df['date'] + date_delta[i]
        projects_df['date-' + str(i + 1)] = projects_df['date'] - date_delta[i]

    # compute total projects within a week, two weeks, and a month for each date and school/zip/city pair
    projects_df = _cnt_wk_bwk_mth_combination(projects_df, 'schoolid')
    projects_df = _cnt_wk_bwk_mth_combination(projects_df, 'school_zip')
    projects_df = _cnt_wk_bwk_mth_combination(projects_df, 'school_city')

    # compute average cost of all project (excluding optional supports) for each school city
    tmp = projects_df.groupby('school_city')['total_price_excluding_optional_support'].agg(np.mean).to_frame(name='ave_proj_cost_school_city')
    tmp.reset_index(inplace=True)
    projects_df = pd.merge(projects_df, tmp, how='left', on='school_city')

    output_df = projects_df[_columns_to_write()]

    # wrtie to csv
    print('writing to csv')
    output_df.to_csv(os.path.join('../Features_csv', 'cnt_bw_wk_mth_combo.csv'), index=False)





