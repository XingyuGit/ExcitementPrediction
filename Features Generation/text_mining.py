import os
import sys
import numpy as np
import pandas as pd 
import re
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append('..')
import import_data as im
		
def get_idf(df, var):
	"""
	get inverse document frequency with all not none data
	"""
	idf = TfidfVectorizer(min_df=2, use_idf=1, smooth_idf=1, sublinear_tf=1, ngram_range=(1,2), token_pattern=r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b", norm='l2')
	idf.fit(df[var][(df["group"]=="train") | (df["group"]=="val") | (df["group"]=="test")])
	return idf

def cross_validate(df, var,idf, seg_name):
	"""
	do cross validation with 1/10 data as train data and 9/10 as test data
	"""
	for i in range(0,10):            
		train_temp = (df["group"]=="train") & ((df['r']<i*0.1) | (df['r']>=(i+1)*0.1))   
		test_temp = (df["group"]=="train") & ((df['r']>=i*0.1) & (df['r']<(i+1)*0.1))           
		df[seg_name][test_temp] = get_predications(df, idf, df[var][train_temp], df[var][test_temp], df["y"][train_temp].values)                                       
def get_predications(df, idf, train_set, test_set,target_values):
	"""
	get predication using liner regression model
	"""
	m_train = idf.transform(train_set)
	m_test=idf.transform(test_set)
	# print m_train
	# print target_values
	lm = SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)
	lm.fit(m_train, target_values)
	return lm.predict_proba(m_test)[:,1]

def get_length(df, list_of_text_vars):
	"""
	replace anomalious character and get the length for each variable
	"""
	for var in list_of_text_vars:
		df.loc[pd.isnull(df[var]), var] = ""
		df[var]=df[var].apply(lambda x:re.sub("\t|\n|\r|\W|\s{2,}|[^A-Za-z0-9\']", " ", x)) 
		df[var+"_length"] = df[var].apply(lambda x:len(x.split()))  

if __name__ == '__main__':
	# if path is not specified, default is 'Data'
	path = sys.argv[1] if len(sys.argv) > 1 else '../Data'
	filepath_1 = os.path.join('../Features_csv', 'essay_pred_val_1.csv')
	filepath_2 = os.path.join('../Features_csv', 'essay_pred_val_2.csv')
	df=im.get_essays_df(path)
	projects_df = im.get_projects_df(path)
	projects_df = projects_df[projects_df['group']!='none']
	outcomes_df = im.get_outcomes_df(path)
	df = pd.merge(df, outcomes_df, how = 'left', on = 'projectid')
	df = pd.merge(df, projects_df, how = 'inner', on = 'projectid')
	df["y"] = 0
	df["y"][df["is_exciting"]=="t"] = 1
	list_of_text_vars=["title", "short_description", "need_statement", "essay"]
	get_length(df, list_of_text_vars)
	#store a random number between 0 and 1 for each row
	df["r"] = np.random.uniform(0,1,size=len(df))
	for var in list_of_text_vars:
		seg_name=var+"_pred_segmental"
		#the segmental predication is 0.0 if it's not in train set
		df[seg_name]=0.0
		idf=get_idf(df, var)
		df[var+"_pred"]=get_predications(df, idf, df[var][df["group"]=="train"], df[var], df["y"][df["group"]=="train"].values)	
		cross_validate(df, var, idf, seg_name)
	to_write_list_1=["projectid","essay_pred","essay_pred_segmental","essay_length","title_pred","title_pred_segmental","title_length"]
	to_write_list_2=["projectid","short_description_pred","short_description_pred_segmental","need_statement_pred","need_statement_pred_segmental"]
	df[to_write_list_1].to_csv(filepath_1, index=False)
	df[to_write_list_2].to_csv(filepath_2, index=False)
	del df
