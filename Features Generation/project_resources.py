__author__ = 'Niyan Ying'

#
#   Features # 13,14,36,43,44,52-58
#

import sys
import os
import pandas as pd
import datetime
import numpy as np
sys.path.append('..')
import import_data

# feature: xx_item_amount_total -- total amount of money a project needs for xx-type resource
def _item_amount_total(df,df2,key,type):   
    tmp = df[df['project_resource_type']==type].groupby('projectid')['item_price_total'].agg(np.sum).to_frame(name=key)
    tmp.reset_index(inplace=True)
    merged_df = pd.merge(df2,tmp,how='left',on='projectid')
    merged_df.loc[pd.isnull(merged_df[key]), key] = 0
    return merged_df
    
if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../Data'
    projects_df = import_data.get_projects_df(path)
    resource_df = import_data.get_resources_df(path)

# feature 43, 44
    projects_df = projects_df[['projectid','resource_type','eligible_double_your_impact_match', 'eligible_almost_home_match','date_posted', 'group']]
# feature 47
    resource_df = resource_df[['project_resource_type','item_unit_price','item_quantity','projectid','resourceid']]

# feature 13: 'eligible_double_impace'
    projects_df['eligible_double_impact'] = np.nan
    projects_df.loc[projects_df['eligible_double_your_impact_match']=='t', 'eligible_double_impact'] = 1
    projects_df.loc[projects_df['eligible_double_your_impact_match']=='f', 'eligible_double_impact'] = 0

# feature 14: 'eligible_almost_home'
    projects_df['eligible_almost_home'] = np.nan
    projects_df.loc[projects_df['eligible_almost_home_match']=='t','eligible_almost_home'] = 1
    projects_df.loc[projects_df['eligible_almost_home_match']=='f','eligible_almost_home'] = 0

# feature 36: 'main_resource_type' for a project
    projects_df['main_resource_type'] = 0
    projects_df.loc[projects_df['resource_type']=='Books','main_resource_type'] = 1
    projects_df.loc[projects_df['resource_type']=='Other','main_resource_type'] = 2
    projects_df.loc[projects_df['resource_type']=='Supplies','main_resource_type'] = 3
    projects_df.loc[projects_df['resource_type']=='Technology','main_resource_type'] = 4
    projects_df.loc[projects_df['resource_type']=='Trips','main_resource_type'] = 5
    projects_df.loc[projects_df['resource_type']=='Visitors','main_resource_type'] = 6

# feature 52: 'agg_item_amount_total'-- total amount of resource(measured by money) a project need		
    resource_df['item_price_total'] = resource_df['item_unit_price']*resource_df['item_quantity']	
    tmp = resource_df.groupby('projectid')['item_price_total'].agg(np.sum).to_frame(name='item_amount_total_per_project')							
    tmp.reset_index(inplace=True)
    projects_df = pd.merge(projects_df, tmp, how='left', on='projectid')
    projects_df.loc[pd.isnull(projects_df['item_amount_total_per_project']), 'item_amount_total_per_project'] = 0

# feature 53: 'agg_cnt'-- number of resources a project needs
    tmp1 = resource_df.groupby('projectid').size().to_frame(name='resource_total_per_project')
    tmp1.reset_index(inplace=True)
    projects_df = pd.merge(projects_df, tmp1, how='left', on='projectid')
    projects_df.loc[pd.isnull(projects_df['resource_total_per_project']), 'resource_total_per_project'] = 0
    
# feature 54-57   
    projects_df = _item_amount_total(resource_df, projects_df,'books_item_amount_total','Books')
    projects_df = _item_amount_total(resource_df, projects_df,'other_item_amount_total','Other')
    projects_df = _item_amount_total(resource_df, projects_df,'supplies_item_amount_total','Supplies')
    projects_df = _item_amount_total(resource_df, projects_df,'tech_item_amount_total','Technology')

# feature 58: weekday. convert date_posted into a weekday
    projects_df['weekday'] =  pd.DatetimeIndex(pd.to_datetime(projects_df['date_posted'],'%Y-%m-%d')).weekday

# outputs_df = merge(projects_df,resource_df)  
    project_features_to_write = ['projectid', 'group','eligible_double_impact', 'eligible_almost_home', 'weekday']
    resource_features_to_write = ['projectid','main_resource_type', 'item_amount_total_per_project', 'resource_total_per_project', 'books_item_amount_total', 'other_item_amount_total', 'supplies_item_amount_total', 'tech_item_amount_total']
    
    outcome_df = projects_df[projects_df['group'] != 'none']
    
    print('writing to csv')
    outcome_df[project_features_to_write].to_csv(os.path.join('../Features_csv', 'project_eligibility_orgin.csv'), index=False)
    outcome_df[resource_features_to_write].to_csv(os.path.join('../Features_csv', 'resource_cnt.csv'), index=False)

    """
    # correctness validation
    outcome_df[project_features_to_write][:10000].to_csv(os.path.join('../Features_csv', 'project_eligibility_orgin_test.csv'))
    outcome_df[resource_features_to_write][:10000].to_csv(os.path.join('../Features_csv', 'resource_cnt_test.csv'))
    """