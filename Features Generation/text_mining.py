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
def build_tfidf_models(df, list_of_text_vars):
	y_values = df["y"][df["group"]=="train"].values
	df["r"] = np.random.uniform(0,1,size=len(df))#get len(df) uniform distributed numbers between 0,1
	for var in list_of_text_vars:
		df[var+"_pred_partial"]=0.0
		vectorizer = TfidfVectorizer(min_df=2, use_idf=1, smooth_idf=1, sublinear_tf=1, ngram_range=(1,2), token_pattern=r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b", norm='l2')  
		# print df[var]
		vectorizer.fit(df[var])
		m_all=vectorizer.transform(df[var])
		lm=get_liner_model(vectorizer, df[var][df["group"]=="train"], y_values)
		df[var+"_pred"] = lm.predict_proba(m_all)[:,1]
		for i in range(0,10):            
			train_temp = (df["group"]=="train") & ((df['r']<i*0.1) | (df['r']>=(i+1)*0.1))   
			test_temp = (df["group"]=="train") & ((df['r']>=i*0.1) & (df['r']<(i+1)*0.1)) #filter of test set;              
			lm_temp=get_liner_model(vectorizer, df[var][train_temp], df["y"][train_temp].values)
			m_test_temp  = vectorizer.transform(df[var][test_temp])                     
			pred_test_temp = lm_temp.predict_proba(m_test_temp)[:,1]
			df[var+"_pred_partial"][test_temp] = pred_test_temp #store the result of last run                                            
			print 'CV: ' + str(i+1)
			print 'AUC (Train_Test): ' + str(metrics.roc_auc_score(df['y'][test_temp],pred_test_temp))  #the test score;
def get_liner_model(vectorizer, train_set, target_values):
	m_train = vectorizer.transform(train_set)
	lm = SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)
	lm.fit(m_train, target_values)
	return	lm
   #get the number of words of text
def get_length(string):
	return len(string.split())
   #replace spaces and anomalies to single space     
def flatten_text(string):
	string = re.sub("\t|\n|\r|\W|\s{2,}", " ", string)   
	# string = string.lower()
	return string.strip()

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
	df.loc[df["is_exciting"]=="t", "y"] = 1
	list_of_text_vars=["title", "short_description", "need_statement", "essay"]
	for var in list_of_text_vars:
		df.loc[pd.isnull(df[var]), var] = ""
		df[var].apply(flatten_text)
		df[var+"_length"] = df[var].apply(get_length)
	build_tfidf_models(df, list_of_text_vars)

	to_write_list_1=["projectid","essay_pred","essay_pred_partial","essay_length","title_pred","title_pred_partial","title_length"]
	to_write_list_2=["projectid","short_description_pred","short_description_pred_partial","need_statement_pred","need_statement_pred_partial",]
	df[to_write_list_1].to_csv(filepath_1, index=False)
	df[to_write_list_2].to_csv(filepath_2, index=False)

