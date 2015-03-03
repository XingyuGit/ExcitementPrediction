import os
import pandas as pd


def __read_file(path, fn):
    print('Reading file: ' + fn + '...')
    data_df = pd.read_csv(os.path.join(path, fn))
    return data_df


def get_outcomes_df(path):
    fn = 'outcomes.csv'
    outcomes_df = __read_file(path, fn)
    outcomes_df['y'] = 0
    outcomes_df.loc[outcomes_df['is_exciting'] == 't', 'y'] = 1
    return outcomes_df


def get_projects_df(path):
    fn = 'projects.csv'
    projects_df = __read_file(path, fn)
    # only data after 2010-4-1 will be used
    # for complete model building, training data  is from 2010-4-1 to 2014-1-1 (exclusive)
    # for optimization, use validation data to optimize performance
    # for prediction, using testing data (after 2014-4-1)
    projects_df['group'] = 'train'
    projects_df.loc[projects_df['date_posted'] < '2010-04-01', 'group'] = 'none'
    projects_df.loc[projects_df['date_posted'] >= '2013-01-01', 'group'] = 'valid'
    projects_df.loc[projects_df['date_posted'] >= '2014-01-01', 'group'] = 'test'
    return projects_df


def get_essays_df(path):
    fn = 'essays.csv'
    essays_df = __read_file(path, fn)
    return essays_df


def get_resources_df(path):
    fn = 'resources.csv'
    resources_df = __read_file(path, fn)
    return resources_df


