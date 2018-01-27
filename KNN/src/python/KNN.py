import numpy as np
import operator
import pandas as pd


def L1(x1, x2):
    return np.sum(np.abs(x1-x2))

def L2(x1, x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

def Linf(x1, x2):
    return np.max(x1-x2)

def cos_dis(x1, x2):
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    if(norm1!=0 and norm2!=0):
        dis = dot/(norm1*norm2)
        norm = -1*(dis-1)
    else:
        norm = 0
    return norm

def get_neighbors(x1, x2):
    size1 = len(x1)
    size2 = len(x2)
    dis = np.zeros(size1*size2).reshape((size2, size1))
    for i in range(size2):
        for j in range(size1):
            dis[i][j] =cos_dis(x2[i], x1[j])
    ind = np.argsort(dis, axis=1, kind='quicksort', order=None)
    return ind, dis

def knn_clssify(train, test, k, train_lables):
    ind, dis = get_neighbors(train, test)
    vnum = ind.shape[0]
    result = np.zeros(vnum).astype(str)
    for i in range(vnum):
        ldict = {}
        for j in range(k):
            lb = train_lables[ind[i][j]]
            d = dis[i][ind[i][j]]
            if(d==0):
                ldict[lb] = 0xfffff
                break
            else:     
                # discard the neighbor with the min similarity (it's useless)     
                if(d<1):      
                    ldict[lb] = ldict.get(lb,0) + 1/d  
        # descending order
        result[i] = sorted(ldict.items(),key=operator.itemgetter(1),reverse=True)[0][0]
    return result


def knn_regression(train, test, k, train_lables):
    ind, dis = get_neighbors(train, test)
    avg_dis = np.average(dis)
    vnum = ind.shape[0]
    emotions = train_lables.shape[1]
    result = np.zeros((vnum,emotions))
    for e in range(emotions):
        for i in range(vnum):
            average = 0
            dist_sum = 0
            for j in range(k):
                if(dis[i][ind[i][j]]==0):
                    average = train_lables[ind[i][j]][e]
                    break
                else:
                    d = dis[i][ind[i][j]]
                    if(d<avg_dis):
                        e_value = train_lables[ind[i][j]][e]/d
                    else:
                        e_value = train_lables[ind[i][j]][e]/(d*2)
                    average += e_value
            result[i][e] = average
    for i in range(vnum):
        result[i] = result[i]/np.sum(result[i])
    return result    
