import os
import pandas as pd

__folder = "Data"


def __read_file(fn):
    print('Reading file: ' + fn + '...')
    data_df = pd.read_csv(os.path.join(__folder), fn)
    return data_df


def get_outcomes_df():
    fn = 'outcomes.csv'
    outcomes_df = __read_file(fn)
    return outcomes_df


def get_projects_df():
    fn = 'projects.csv'
    projects_df = __read_file(fn)
    return projects_df


def get_essays_df():
    fn = 'essays.csv'
    essays_df = __read_file(fn)
    return essays_df


def get_resources_df():
    fn = 'resources.csv'
    resources_df = __read_file(fn)
    return resources_df


