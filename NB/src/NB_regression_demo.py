from __future__ import absolute_import

import numpy as np
import operator
import pandas as pd
import math
from ..preprocessing.text_processing import *
from NB import nb_regression_train
from NB import nb_regression

def main():
    train_file = 'train_set.csv'
    train_data = pd.read_csv(train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(valid_file)

    txt_data = train_data['Words (split by space)']
    txt_data = txt_data.str.split().values
    data_dict = get_word_dict(txt_data) 
    tf_mat, line_len = tf(txt_data,data_dict)
    size = len(txt_data)
    valid_data_x = valid_data['Words (split by space)']
    valid_data_x = valid_data_x.str.split().values
    
    valid_oh = one_hot(valid_data_x, data_dict)

    train_lables = train_data.loc[:,['anger','disgust','fear','joy','sad','surprise']].values
    
    likelihood = nb_regression_train(tf_mat, line_len, train_lables)
    result = nb_regression(likelihood, valid_oh, train_lables)

    valid_result = 'nb_valid_result.csv'
    with open(valid_result, 'wb') as f:
        np.savetxt(f, result, delimiter = ',', fmt = '%.4f')


    

if __name__ == '__main__':
    main()