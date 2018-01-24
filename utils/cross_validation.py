import numpy as np
import pandas as pd
from random import randrange

# k folds cross validation
def split_dataset(data, k):
    sdata = []
    temp_data = data
    data_size = data.shape[0]
    fold_size = data_size // k
    for i in range(k):
        sub = []
        while len(sub) < fold_size:   
            index = randrange(temp_data.shape[0])
            sub.append(temp_data[index])
            temp_data = np.delete(temp_data, index, axis = 0)
        sdata.append(np.array(sub))
    return np.array(sdata)

if __name__ == '__main__':
    data = pd.read_csv('.\\my_train.csv',header=None).values
    sdata = split_dataset(data,3)
    print(sdata)
    print(sdata.reshape(data.shape[0],data.shape[1]))

