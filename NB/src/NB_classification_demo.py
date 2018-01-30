from __future__ import absolute_import

import numpy as np
import operator
import pandas as pd
import math
from ..preprocessing.text_processing import *
from NB import nb_clssify_train
from NB import nb_clssify


def main():
    train_file = 'train_set.csv'
    train_data = pd.read_csv(train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(valid_file)

    txt_data = train_data['Words (split by space)']
    txt_data = txt_data.str.split().values
    data_dict = get_word_dict(txt_data) 
    
    size = len(txt_data)
    valid_data_x = valid_data['Words (split by space)']
    valid_data_x = valid_data_x.str.split().values

    valid_oh = one_hot(valid_data_x, data_dict)

    train_lables = train_data['label'].values

    prior, likelihood = nb_clssify_train(txt_data, train_lables, data_dict, 'multinomial')
    result = nb_clssify(prior, likelihood, valid_oh)

    gt = valid_data['label'].values
    corrct = 0
    vnum = result.shape[0]
    for i in range(vnum):
        if(result[i]==gt[i]):
            corrct += 1

    pre = corrct/vnum
    print('classification presicion', pre)    

if __name__ == '__main__':
    main()