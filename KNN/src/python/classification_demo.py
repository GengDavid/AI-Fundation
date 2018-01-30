from __future__ import absolute_import

import numpy as np
import operator
import pandas as pd
from ..preprocessing.text_processing import *


def classify(train_data, valid_data):
    txt_data = train_data['Words (split by space)']

    txt_data = txt_data.str.split().values
    data_dict = get_word_dict(txt_data) 
    tf_mat = tf(txt_data,data_dict)
    idf_vec = idf(txt_data, data_dict)
    tf_idf = tf_mat*idf_vec
    
    for i in range(tf_idf.shape[0]):
        l2_norm = np.sum(np.square(tf_idf[i]))
        if(l2_norm!=0):
            tf_idf[i] = (tf_idf[i])/l2_norm
    
    size = len(txt_data)

    
    valid_data_x = valid_data['Words (split by space)']
    valid_data_x = valid_data_x.str.split().values

    valid_tf = tf(valid_data_x, data_dict)
    valid_idf = idf(valid_data_x, data_dict)
    valid_tfidf = valid_tf*valid_idf

    for i in range(valid_tfidf.shape[0]):
        l1_norm = np.sum(np.abs(valid_tfidf[i]))
        if(l1_norm!=0):
            valid_tfidf[i] = (valid_tfidf[i])/l1_norm

    
    train_lables = train_data['label'].values

    pre = np.zeros(25)
    for k in range(26):
        if(k==0):
            continue
        result = knn_clssify(tf_idf, valid_tfidf, k, train_lables)

        gt = valid_data['label'].values
        corrct = 0
        vnum = result.shape[0]
        for i in range(vnum):
            if(result[i]==gt[i]):
                corrct += 1
        pre[k-1] = corrct/vnum
        # print('classification presicion', pre)
    write_path = './result/'
    cl_result = 'classification.csv'
    with open(write_path+cl_result, 'wb') as f:
        np.savetxt(f, pre, delimiter = ',', fmt = '%.6f')    


def main():
	train_file = 'train_set.csv'
    train_data = pd.read_csv(train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(valid_file)

	classify(train_data, valid_data)    

if __name__ == '__main__':
    main()