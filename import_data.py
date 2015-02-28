import os
import pandas as pd

folder = "Data"



def read_file(fn):
    print('Reading file: ' + fn + '...')
    data_df = pd.read_csv(os.path.join(folder), fn)
    return data_df


def get_outcome_df():
    fn = 'outcomes.csv'
    outcome_df = read_file(fn)
    return outcome_df


def get_project_df():
    fn = 'projects.csv'
    projects_df = read_file(fn)
    return projects_df


def get_essays_df():
    fn = 'essays.csv'
    essays_df = read_file(fn)
    return essays_df


def get_resources_df():
    fn = 'resources.csv'
    resources_df = read_file(fn)
    return resources_df


