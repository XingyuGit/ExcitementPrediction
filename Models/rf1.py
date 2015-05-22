__author__ = 'TerryChen'

#
#   Features : 2-39
#

import os
import sys
import model_train_predict as model
import import_data

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

    input_files = [fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8]
    output_file = 'rf1_predict.csv'
    features_fn = 'rf1_features.txt'

    features = read_features(features_fn)
    gbm = model.train_mode(model='rf', features=features)
    gbm.set_model_parameters(number_trees=5000, max_leaf_nodes=10, max_features=2)
    gbm.train_and_predict(input_files=input_files, output_fn=output_file)






