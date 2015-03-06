import sys
import os
import pandas as pd
from pandasql import sqldf
import numpy as np
import datetime
sys.path.append('..')
import import_data

def _get_average(df,frequency,valueKey,countKey,name):
    df[name] = pd.rolling_sum(df[valueKey],window=frequency) / pd.rolling_sum(df[countKey],window=frequency)
    df.ix[:frequency-1, name] = df.ix[:frequency-1, valueKey] / df.ix[:frequency-1, countKey]


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../Data'
    projects_df = import_data.get_projects_df(path)

    #feature 40,41,46
    projects_df = projects_df[['projectid','school_metro', 'teacher_ny_teaching_fellow', 'students_reached', 'poverty_level', 'primary_focus_area', 'total_price_excluding_optional_support','date_posted']]
    
    #feature 42
    projects_df.loc[pd.isnull(projects_df['students_reached']),'students_reached'] = 0

    ### Convert Category Data into numerical: Xuhui Chen
    #
    projects_df['metro'] = 0
    projects_df.loc[projects_df['school_metro'] == 'urban', 'metro'] = 1
    projects_df.loc[projects_df['school_metro'] == 'suburban', 'metro'] = 2
    projects_df.loc[projects_df['school_metro'] == 'rural', 'metro'] = 3

    #
    projects_df['primary_subject'] = 0
    projects_df.loc[projects_df['primary_focus_area'] == 'Literacy & Language', 'primary_subject'] = 1
    projects_df.loc[projects_df['primary_focus_area'] == 'History & Civics', 'primary_subject'] = 2
    projects_df.loc[projects_df['primary_focus_area'] == 'Math & Science', 'primary_subject'] = 3
    projects_df.loc[projects_df['primary_focus_area'] == 'Health & Sports', 'primary_subject'] = 4
    projects_df.loc[projects_df['primary_focus_area'] == 'Applied Learning', 'primary_subject'] = 5
    projects_df.loc[projects_df['primary_focus_area'] == 'Music & The Arts', 'primary_subject'] = 6
    projects_df.loc[projects_df['primary_focus_area'] == 'Special Needs', 'primary_subject'] = 7

    #
    projects_df['teacher_ny_fellow'] = 1
    projects_df.loc[projects_df['teacher_ny_teaching_fellow'] == 'f', 'teacher_ny_fellow'] = 0
    
    #feature 45
    projects_df['poverty'] = 0
    projects_df.loc[projects_df['poverty_level'] == 'moderate poverty', 'poverty'] = 1
    projects_df.loc[projects_df['poverty_level'] == 'high poverty','poverty'] = 2
    projects_df.loc[projects_df['poverty_level'] == 'highest poverty','poverty'] = 3
    projects_df.loc[projects_df['poverty_level'] == 'low poverty', 'poverty'] = 4

    #feature 75
    projects_df['price_per_student'] = projects_df['total_price_excluding_optional_support']/projects_df['students_reached']
    projects_df.loc[np.isinf(projects_df['price_per_student']),'price_per_student'] = 0

    
    #feature 82,83,84
    projects_df['predM1'] = 0
    projects_df.ix[::2,'predM1'] = 1
    pysqldf = lambda q: sqldf(q,globals())
    temp_df = pysqldf("select count(projectid) as numOfProjects,date_posted ,sum(predM1) as sumOfPredM1 from projects_df where date_posted >= '2010-09-01' group by date_posted")
    _get_average(temp_df,14,'sumOfPredM1','numOfProjects','average_biweekly_predM1')
    _get_average(temp_df,30,'sumOfPredM1','numOfProjects','average_monthly_predM1')
    _get_average(temp_df,61,'sumOfPredM1','numOfProjects','average_bimonthly_predM1')
    temp_df = temp_df[['date_posted','average_biweekly_predM1','average_monthly_predM1','average_bimonthly_predM1']]
    projects_df = pd.merge(projects_df, temp_df, how='left', on='date_posted')


    projects_df = projects_df[['projectid','metro', 'teacher_ny_fellow', 'students_reached', 'poverty', 'primary_subject','price_per_student','average_biweekly_predM1','average_monthly_predM1','average_bimonthly_predM1']]
    projects_df.to_csv(os.path.join('../Features_csv', 'exciting_project_rolling_average.csv'), index=False)
