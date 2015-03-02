import pandas as pd 
import numpy as np
from import_data import *
from time import *

def __quantify(df, list_of_columns):
    """
    Transform the categorical data to numeric data (f -> 0, t -> 1):
        Add new column x_cnt for each column x
    """
    for col in list_of_columns:
        new_col = col + '_cnt'
        df[new_col] = 0
        # df[new_col][df[col]=='t'] = 1
        df.loc[df[col]=='t', new_col] = 1

def __acc_cnt(df, list_of_vars, list_of_cnt_cols):
    df['one']  = 1
    for var in list_of_vars:
        df_sorted = df.sort([var, 'date_posted'])
        df_grouped = df.groupby(var)

        acc_cnt_col = 'cum_cnt_for_' + var
        df_sorted[acc_cnt_col] = df_grouped['one'].cumsum() - df_sorted['one']
        df[acc_cnt_col] = df_sorted[acc_cnt_col].sort_index()

        for cnt_col in list_of_cnt_cols:
            cumsum_col = cnt_col + '_acc_for_' + var
            rate_col = cnt_col + '_rate_for_' + var
            df_prev_cumsum = df_grouped[cnt_col].cumsum() - df_sorted[cnt_col]
            df_prev_rate = df_prev_cumsum / df_sorted[acc_cnt_col]
            df_prev_rate[np.isinf(df_prev_rate)] = 0
            df[cumsum_col] = df_prev_cumsum.sort_index()
            df[rate_col] = df_prev_rate.sort_index()
    del df['one']

projects_df = get_projects_df()
outcomes_df = get_outcomes_df()

list_of_columns = ['is_exciting', 'at_least_1_teacher_referred_donor', 'fully_funded',
    'at_least_1_green_donation', 'great_chat', 'three_or_more_non_teacher_referred_donors',
    'one_non_teacher_referred_donor_giving_100_plus', 'donation_from_thoughtful_donor']

# list_of_columns = ['is_exciting']
list_of_cnt_cols = [x + '_cnt' for x in list_of_columns]
list_of_vars = ['teacher_acctid']

tick = time();
__quantify(outcomes_df, list_of_columns)
print "elapsed time: " + str((time() - tick));
# print outcomes_df[:5]

df = pd.merge(projects_df, outcomes_df, how = 'left', on = 'projectid')
__acc_cnt(df, list_of_vars, list_of_cnt_cols)

print df[df['group'] != 'test'][['projectid', 'group', 'teacher_acctid', 'is_exciting', 'is_exciting_cnt', 'is_exciting_cnt_acc_for_teacher_acctid', 'cum_cnt_for_teacher_acctid', 'is_exciting_cnt_rate_for_teacher_acctid']].head()
df[df['group']=='train'][:500].to_csv('history.csv')


