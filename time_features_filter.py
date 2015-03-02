__author__ = 'TerryChen'

#
#   Features # 20 - 24, 7
#

import sys
import os
import import_data
import pandas as pd
import datetime
import numpy as np

def _timedelta(d):
    return datetime.timedelta(days=d)


def _cnt_wk_bwk_mth_combination(df, key):
    print('Compute count for {} and date pair for a week, two weeks, and a month'.format(key))

    key_abbr = key
    if key == 'school_zip':
        key_abbr = 'zip'
    elif key == 'school_city':
        key_abbr = 'city'

    tmp = df.groupby(['date', key]).size().to_frame(name='cnt_day_{}'.format(key))
    tmp.reset_index(inplace=True)
    df = pd.merge(df, tmp, how= 'left', on=[key, 'date'])

    for i in range(6):
        tmp.columns = ['date+{}'.format(i + 1), key, 'cnt_day_{}+{}'.format(key_abbr, i + 1)]
        df = pd.merge(df, tmp, how= 'left', on= [key, 'date+{}'.format(i + 1)])
        df['cnt_day_{}+{}'.format(key_abbr, i + 1)][df['cnt_day_{}+{}'.format(key_abbr, i + 1)].apply(lambda c: pd.isnull(c))] = 0

        tmp.columns = ['date-{}'.format(i + 1), key, 'cnt_day_{}-{}'.format(key_abbr, i + 1)]
        df = pd.merge(df, tmp, how= 'left', on= [key, 'date-{}'.format(i + 1)])
        df['cnt_day_{}-{}'.format(key_abbr, i + 1)][df['cnt_day_{}-{}'.format(key_abbr, i + 1)].apply(lambda c: pd.isnull(c))] = 0

    tmp = df.groupby(['yearmonth', key]).size().to_frame(name='cnt_mth_{}'.format(key_abbr))
    tmp.reset_index(inplace=True)
    df = pd.merge(df, tmp, how='left', on=[key, 'yearmonth'])

    # sum for total counts
    cnt = df['cnt_day_' + key]
    for i in range(3):
        cnt += df['cnt_day_{}+{}'.format(key_abbr, i + 1)]
        cnt += df['cnt_day_{}-{}'.format(key_abbr, i + 1)]
    df['cnt_wk_' + key_abbr] = cnt

    cnt = df['cnt_day_' + key]
    for i in range(6):
        cnt += df['cnt_day_{}+{}'.format(key_abbr, i + 1)]
        cnt += df['cnt_day_{}-{}'.format(key_abbr, i + 1)]
    df['cnt_bwk_' + key_abbr] = cnt

    return df



if __name__ == '__main__':
    # if path is not specified, default is 'Data'
    path = sys.argv[1] if len(sys.argv) > 1 else 'Data'
    projects_df = import_data.get_projects_df(path)
    projects_df = projects_df[['projectid', 'schoolid', 'date_posted', 'school_city', 'school_zip', 'total_price_excluding_optional_support']]

    # delta of date
    date_delta = [_timedelta(1), _timedelta(2), _timedelta(3), _timedelta(4), _timedelta(5), _timedelta(6)]
    projects_df['date'] = projects_df['date_posted'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d') if not pd.isnull(d) else d)
    projects_df['yearmonth'] = projects_df['date_posted'].apply(lambda d: datetime.datetime.strptime(d[:7], '%Y-%m') if not pd.isnull(d) else d)

    # introduce columns of dates from current_date + 6 to current_date - 6
    for i in range(6):
        projects_df['date+' + str(i + 1)] = projects_df['date'] + date_delta[i]
        projects_df['date-' + str(i + 1)] = projects_df['date'] - date_delta[i]

    # compute total projects within a week, two weeks, and a month for each date and school/zip/city pair
    projects_df = _cnt_wk_bwk_mth_combination(projects_df, 'schoolid')
    projects_df = _cnt_wk_bwk_mth_combination(projects_df, 'school_zip')
    projects_df = _cnt_wk_bwk_mth_combination(projects_df, 'school_city')

    # compute average cost of all project (excluding optional supports) for each school city
    tmp = projects_df.groupby('school_city')['total_price_excluding_optional_support'].agg(np.mean).to_frame(name='price_school_city')
    tmp.reset_index(inplace=True)
    projects_df = pd.merge(projects_df, tmp, how='left', on='school_city')

    output_df = projects_df[['projectid', 'price_school_city', 'cnt_wk_schoolid', 'cnt_bwk_schoolid', 'cnt_mth_schoolid', 'cnt_wk_zip', 'cnt_bwk_zip', 'cnt_mth_zip', 'cnt_wk_city', 'cnt_bwk_city', 'cnt_mth_city']]

    output_df.to_csv(os.path.join('Features_csv', 'cnt_bw_wk_mth_combo.csv'), index=False)





