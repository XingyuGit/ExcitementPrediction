__author__ = 'ChristinaSun'

#
# Features # 8-13, 27, 28, 30
#

import os
import sys
import numpy as np
import pandas as pd
import random
sys.path.append('..')
import import_data

if __name__ == '__main__':
    #if path is not specified, default is 'Data'
    path = sys.argv[1] if len(sys.argv) > 1 else '../Data'
    projects_df = import_data.get_projects_df(path)
    resource_df = import_data.get_resources_df(path)

    projects_df = projects_df[
        ['projectid','schoolid', 'school_city','school_latitude', 'school_longitude', 'teacher_prefix', 'teacher_teach_for_america',
         'students_reached', 'total_price_including_optional_support', 'total_price_excluding_optional_support']]
    resource_df = resource_df[['project_resource_type', 'item_quantity', 'projectid', 'resourceid']]

    #feature 10: teacher_gender
    projects_df['teacher_gender'] = np.nan
    projects_df.loc[projects_df['teacher_prefix'] == '', 'teacher_gender'] = 0
    projects_df.loc[projects_df['teacher_prefix'] == 'Dr.', 'teacher_gender'] = 1
    projects_df.loc[projects_df['teacher_prefix'] == 'Mr.', 'teacher_gender'] = 2
    projects_df.loc[projects_df['teacher_prefix'] == 'Mr. & Mrs.', 'teacher_gender'] = 3
    projects_df.loc[projects_df['teacher_prefix'] == 'Mrs.', 'teacher_gender'] = 4
    projects_df.loc[projects_df['teacher_prefix'] == 'Ms.', 'teacher_gender'] = 5

    #feature 11: teach_in_america
    projects_df['teach_in_america'] = np.nan
    projects_df.loc[projects_df['teacher_teach_for_america'] == 'f', 'teach_in_america'] = 0
    projects_df.loc[projects_df['teacher_teach_for_america'] == 't', 'teach_in_america'] = 1

    #feature 12: students_cnt
    projects_df['students_cnt'] = projects_df['students_reached']
    projects_df.loc[pd.isnull(projects_df['students_reached']), 'students_cnt'] = 1

    #feature 13: books_cnt
    tmp = resource_df[resource_df['project_resource_type'] == 'Books'].groupby('projectid')['item_quantity'].agg(np.sum).to_frame(name='books_cnt')
    tmp.reset_index(inplace=True)
    projects_df = pd.merge(projects_df, tmp, how='left', on='projectid')
    projects_df.loc[pd.isnull(projects_df['books_cnt']), 'books_cnt'] = 0

    #feature 30: optional_support
    projects_df['optional_support'] = projects_df['total_price_including_optional_support'] - projects_df['total_price_excluding_optional_support']

    #feature 27: avg_opt_sup_school_city
    tmp = projects_df.groupby('school_city')['optional_support'].agg(np.mean).to_frame(name='ave_opt_sup_school_city')
    tmp.reset_index(inplace=True)
    projects_df = pd.merge(projects_df, tmp, how='left', on='school_city')

    #feature 28: avg_price_excl_school
    tmp = projects_df.groupby('schoolid')['total_price_excluding_optional_support'].agg(np.mean).to_frame(name='ave_price_excl_school')
    tmp.reset_index(inplace=True)
    projects_df = pd.merge(projects_df, tmp, how='left', on='schoolid')

    features_to_write = ['projectid', 'school_latitude', 'school_longitude', 'teacher_gender', 'teach_in_america', 'students_cnt', 'total_price_excluding_optional_support', 'books_cnt', 'ave_opt_sup_school_city', 'ave_price_excl_school', 'optional_support']

    outcome_df = projects_df[features_to_write]

    print('writing to csv')
    outcome_df.to_csv(os.path.join('../Features_csv', 'general_plus_opt_support_stat.csv'), index=False)


