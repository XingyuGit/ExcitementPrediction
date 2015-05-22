__author__ = "Xingyu Zhou"

import os
import sys
import pandas as pd 
sys.path.append('..')

def _select(in_file, out_file, list_of_columns):
    print 'input from: ' + in_file + " ..."
    df = pd.read_csv(in_file)
    print 'output to: ' + out_file + " ..."
    return df.to_csv(out_file, columns=list_of_columns, index=False)

def _list_of_columns():
    list_of_columns = ['projectid', 'group', 'y',
    'teacher_acctid_at_least_1_teacher_referred_donor_cumrate',
    'school_county_at_least_1_teacher_referred_donor_cumrate',
    'school_county_donation_from_thoughtful_donor_cumrate',
    'teacher_acctid_great_chat_cumrate',
    'schoolid_is_exciting_cumcredrate_cap',
    'school_district_great_chat_cumrate',
    'schoolid_at_least_1_green_donation_cumrate',
    'schoolid_gapdays',
    'teacher_acctid_gapdays',
    'school_county_at_least_1_green_donation_cumcredrate',
    'teacher_acctid_is_teacher_acct_cumrate',
    'schoolid_great_chat_cumrate',
    'school_county_great_chat_cumcredrate',
    'school_district_at_least_1_green_donation_cumcredrate',
    'school_county_is_exciting_cumrate',
    'teacher_acctid_at_least_1_green_donation_cumcredrate',
    'school_district_is_exciting_cumcredrate',
    'school_county_donation_from_thoughtful_donor_cumcnt',
    'schoolid_three_or_more_non_teacher_referred_donors_cumcredrate',
    'school_county_cumcnt',
    'school_district_three_or_more_non_teacher_referred_donors_cumrate',
    'teacher_acctid_donation_total_cumcredrate',
    'schoolid_at_least_1_teacher_referred_donor_cumcredrate']
    return list_of_columns

whole_filepath = os.path.join('../Features_csv', 'history_stat.csv')
refine_filepath = os.path.join('../Features_csv', 'selected_history_stat.csv')
_select(whole_filepath, refine_filepath, _list_of_columns())

