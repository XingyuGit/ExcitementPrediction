__author__ = 'Xingyu Zhou'

import os
import sys
import model_train_predict as model
import import_data
import pandas as pd

def read_features(read_fn):
    f = open(read_fn, 'r')
    features = []
    for line in f:
        feature = line.strip()
        features.append(feature)
    f.close()
    return features

if __name__ == '__main__':
    fn1 = 'general_plus_opt_support_stat.csv'
    fn2 = 'project_eligibility_orgin.csv'
    fn3 = 'adjusted_attributes.csv'
    fn4 = 'resource_cnt.csv'
    fn5 = 'essay_pred_val_3.csv'
    fn6 = 'refined_history_stat.csv'
    fn7 = 'cnt_bw_wk_mth_combo.csv'
    fn8 = 'prev_comprisons.csv'
    fn9 = 'exciting_project_rolling_average.csv'

    input_files = [fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9]
    output_file = 'total_predict_{}.csv'
    features_fn = 'total_features.txt'

    features = read_features(features_fn)
    m = model.train_mode(model='gbm', features=features)
    m.set_model_parameters(number_trees=650, learning_rate=0.1, min_sample_split=100, max_leaf_nodes=2)
    m.train_and_predict(input_files=input_files, output_fn=output_file.format("gbm"))

    m = model.train_mode(model='et', features=features)
    m.set_model_parameters(number_trees=3000, max_leaf_nodes=100, max_features=2)
    m.train_and_predict(input_files=input_files, output_fn=output_file.format("et"))

    m = model.train_mode(model='rf', features=features)
    m.set_model_parameters(number_trees=650, max_leaf_nodes=10, max_features=2)
    m.train_and_predict(input_files=input_files, output_fn=output_file.format("rf"))

    gbm = pd.read_csv(os.path.join('../Prediction', 'total_predict_gbm.csv'))
    et = pd.read_csv(os.path.join('../Prediction', 'total_predict_et.csv'))
    rf = pd.read_csv(os.path.join('../Prediction', 'total_predict_rf.csv'))

    m = gbm.copy()
    m['is_exciting'] = 0.45 * gbm['is_exciting'] + 0.45 * et['is_exciting'] + 0.1 * rf['is_exciting']
    m.to_csv(os.path.join('../Prediction', 'total_predict_combine.csv'), index=False)



