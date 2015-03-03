#!/usr/bin/env python
# coding:utf-8

"Re-extract data from csv"

__author__ = "Xingyu Zhou"

import sys
import pandas as pd

def _select(filepath, list_of_columns):
    df = pd.read_csv(filepath)
    return df[list_of_columns]

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../Features_csv/project_history.csv'
    list_of_columns = ['projectid', 'teacher_acctid_gapdays', 'schoolid_gapdays', 'teacher_acctid_cumcnt',
    'schoolid_cumcnt', 'teacher_acctid_is_exciting_cumcredrate', 'schoolid_is_exciting_cumcredrate_cap',
    'school_district_is_exciting_cumrate', 'school_county_is_exciting_cumrate',
    'schoolid_at_least_1_teacher_referred_donor_cumrate',
    'teacher_acctid_at_least_1_teacher_referred_donor_cumrate',
    'schoolid_fully_funded_cumrate', 'teacher_acctid_fully_funded_cumrate',
    'school_district_fully_funded_cumrate', 'schoolid_at_least_1_green_donation_cumrate',
    'schoolid_great_chat_cumrate', 'teacher_acctid_great_chat_cumrate',
    'schoolid_cumcnt', 'teacher_acctid_cumcnt', 'schoolid_is_teacher_acct_cumrate',
    'teacher_acctid_is_teacher_acct_cumrate', 'schoolid_donation_total_cumcnt',
    'teacher_acctid_donation_total_cumcnt']
    df = _select(path, list_of_columns)
    df.to_csv('../Features_csv/refined_project_history.csv', index=False)