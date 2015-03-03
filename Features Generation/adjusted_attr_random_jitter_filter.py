__author__ = 'TerryChen'

#
#   Features # 2, 3, 4, 5, 6
#

import os
import sys
import numpy as np
import pandas as pd
import random
sys.path.append('..')
import import_data

def _get_adjusted_attribute(df, y, condition, key1, key2=None, key3=None, percentage_all_exciting='percentage_exciting_all', k=5, r=0.5):
    selected_attributes = [y, key1, percentage_all_exciting]
    group_attributes = [key1]
    if not key2 is None:
        selected_attributes.append(key2)
        group_attributes.append(key2)
    if not key3 is None:
        selected_attributes.append(key3)
        group_attributes.append(key2)

    df_selected = df[selected_attributes]
    df_filtered = df_selected.loc[np.logical_and(condition, np.logical_not(pd.isnull(df_selected['y']))), :]
    df_groups = df_filtered.groupby(group_attributes)

    df1 = df_groups.size().to_frame(name='projects_cnt')
    df1.reset_index(inplace=True)

    df2 = df_groups['y'].agg(np.sum).to_frame(name='exciting_cnt')
    df2.reset_index(inplace=True)

    df_selected = pd.merge(df_selected, df1, how='left', on=group_attributes)
    df_selected = pd.merge(df_selected, df2, how='left', on=group_attributes)

    df_selected.loc[pd.isnull(df_selected['projects_cnt']), 'projects_cnt'] = 0
    df_selected.loc[pd.isnull(df_selected['exciting_cnt']), 'exciting_cnt'] = 0

    df_selected.loc[condition, 'projects_cnt'] -= 1
    df_selected.loc[condition, 'exciting_cnt'] -= df_selected.loc[condition, 'y']
    df_selected['expection_exciting'] = df_selected['projects_cnt'] / df_selected['exciting_cnt']
    df_selected['adjusted_y'] = (df_selected['exciting_cnt'] + df_selected[percentage_all_exciting] * k) / (df_selected['projects_cnt'] + k)
    df_selected.loc[pd.isnull(df_selected['expection_exciting']), 'expection_exciting'] = df_selected.loc[pd.isnull(df_selected['expection_exciting']), percentage_all_exciting]
    df_selected.loc[pd.isnull(df_selected['adjusted_y']), 'adjusted_y'] = df_selected.loc[pd.isnull(df_selected['adjusted_y']), percentage_all_exciting]

    random_numbers = [random.random() for x in range(np.sum(condition))]
    df_selected.loc[condition, 'adjusted_y'] *= [(x - 0.5) * r + 1 for x in random_numbers]
    return df_selected['adjusted_y']

if __name__ == '__main__':
    # if path is not specified, default is 'Data'
    path = sys.argv[1] if len(sys.argv) > 1 else '../Data'
    projects_df = import_data.get_projects_df(path)
    outcomes_df = import_data.get_outcomes_df(path)

    attributes = ['projectid', 'group', 'teacher_acctid', 'schoolid', 'school_state', 'primary_focus_subject', 'secondary_focus_subject', 'resource_type', 'y']

    projects_df = projects_df[attributes[:-1]]
    outcomes_df = outcomes_df[[attributes[0], attributes[-1:][0]]]

    data_df = pd.merge(projects_df, outcomes_df, how='left', on='projectid')
    data_df['non_test'] = 0
    data_df.loc[(projects_df['group'] == 'valid') | (projects_df['group'] == 'train'), 'non_test'] = 1

    # percentage of exciting projects in all non-test set ( train + valid )
    data_df['percentage_exciting_all'] = np.mean(data_df['y'])

    # set seed to current system time
    random.seed(None)
    data_df['adj_teacher_acctid'] = _get_adjusted_attribute(data_df, 'y', data_df['non_test']==1, 'teacher_acctid', k=5, r=0.3)
    data_df['adj_schoolid'] = _get_adjusted_attribute(data_df, 'y', data_df['non_test']==1, 'schoolid', k=25, r=0.3)
    data_df['adj_school_state'] = _get_adjusted_attribute(data_df, 'y', data_df['non_test']==1, 'school_state', k=25, r=0.3)
    data_df['adj_subject'] = _get_adjusted_attribute(data_df, 'y', data_df['non_test']==1, 'primary_focus_subject',key2='secondary_focus_subject', k=20, r=0.3)
    data_df['adj_primary_focus_subject_resource_type'] = _get_adjusted_attribute(data_df, 'y', data_df['non_test']==1, 'primary_focus_subject',key2='resource_type', k=25, r=0.3)
    data_df['adj_secondary_focus_subject_resource_type'] = _get_adjusted_attribute(data_df, 'y', data_df['non_test']==1, 'secondary_focus_subject',key2='resource_type', k=25, r=0.3)

    data_df = data_df[['projectid', 'adj_teacher_acctid', 'adj_schoolid', 'adj_school_state', 'adj_subject', 'adj_primary_focus_subject_resource_type', 'adj_secondary_focus_subject_resource_type']]

    # wrtie to csv
    print('writing to csv')
    data_df.to_csv(os.path.join('../Features_csv', 'adjusted_attributes.csv'), index=False)