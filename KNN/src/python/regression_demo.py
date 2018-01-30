from __future__ import absolute_import

import numpy as np
import operator
import pandas as pd
from ..preprocessing.text_processing import *
from KNN import knn_clssify
from KNN import knn_regression


def regression():
    train_file = 'train_set.csv'
    train_data = pd.read_csv(train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(valid_file)

    k = 13

    txt_data = train_data['Words (split by space)']
    txt_data = txt_data.str.split().values

    size = len(txt_data)
    valid_data_x = valid_data['Words (split by space)']
    valid_data_x = valid_data_x.str.split().values

    data_dict = get_word_dict(txt_data)

    tf_mat = tf(txt_data,data_dict)
    idf_vec = idf(txt_data, data_dict)
    tf_idf = tf_mat*idf_vec
    for i in range(tf_idf.shape[0]):
        l1_norm = np.sum(np.abs(tf_idf[i]))
        if(l1_norm!=0):
            tf_idf[i] = (tf_idf[i])/l1_norm

    valid_tf = tf(valid_data_x, data_dict)
    valid_idf = idf(valid_data_x, data_dict)
    valid_tfidf = valid_tf*valid_idf
    for i in range(valid_tfidf.shape[0]):
        dif = np.std(valid_tfidf[i])
        if(dif!=0):
            valid_tfidf[i] = (valid_tfidf[i]-np.average(valid_tfidf[i]))/dif

    train_lables = train_data.loc[:,['anger','disgust','fear','joy','sad','surprise']].values
    
    result = knn_regression(tf_idf, valid_tfidf, k, train_lables)

    valid_result = 'valid_result.csv'
    with open(+valid_result, 'wb') as f:
        np.savetxt(f, result, delimiter = ',', fmt = '%.6f')


def main():
	train_file = 'train_set.csv'
    train_data = pd.read_csv(train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(valid_file)

	regression(train_data, valid_data)    

if __name__ == '__main__':
    main()