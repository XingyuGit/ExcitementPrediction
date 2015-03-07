__author__ = 'TerryChen'

import os
import sys
import pandas as pd

if __name__ == '__main__':
    fn = os.path.join('../Features_csv', 'essay_pred_val_1.csv')
    if not os.path.isfile(fn):
        sys.exit()

    df = pd.read_csv(fn)

    var_text = ['essay', 'title']
    for var in var_text:
        segment = var + '_pred_segmental'
        pred = var + '_pred'
        df[var + '_new_pred'] = df[segment]
        condition = (df[segment] == 0) | pd.isnull(df[segment])
        df.loc[condition, var + '_new_pred'] = df[pred][condition]

    columns_to_write = ['projectid', 'title_new_pred', 'essay_new_pred', 'essay_length', 'essay_cap_len', 'essay_cap_per', 'title_length', 'title_cap_len', 'title_cap_per']
    df = df[columns_to_write]
    df.to_csv(os.path.join('../Features_csv', 'essay_pred_val_3.csv'), index=False)
    sys.exit()