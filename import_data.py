import os
import pandas as pd

__folder = "Data"


def __read_file(fn):
    print('Reading file: ' + fn + '...')
    data_df = pd.read_csv(os.path.join(__folder, fn))
    return data_df


def get_outcomes_df():
    fn = 'outcomes.csv'
    outcomes_df = __read_file(fn)
    outcomes_df['y'] = 0
    outcomes_df['y'][outcome_df['is_exciting'] == 't'] = 1
    return outcomes_df


def get_projects_df():
    fn = 'projects.csv'
    projects_df = __read_file(fn)
    # only data after 2010-4-1 will be used
    # for complete model building, training data  is from 2010-4-1 to 2014-1-1 (exclusive)
    # for optimization, use validation data to optimize performance
    # for prediction, using testing data (after 2014-4-1)
    projects_df['group'] = 'train'
    projects_df['group'][projects_df['date_posted'] < '2010-04-01'] = 'none'
    projects_df['group'][projects_df['date_posted'] >= '2013-01-01'] = 'valid'
    projects_df['group'][projects_df['date_posted'] >= '2014-01-01'] = 'test'
    return projects_df


def get_essays_df():
    fn = 'essays.csv'
    essays_df = __read_file(fn)
    return essays_df


def get_resources_df():
    fn = 'resources.csv'
    resources_df = __read_file(fn)
    return resources_df


