import numpy as np
import operator
import pandas as pd

def get_word_dict(data_list):
    data_dict = {}
    lines = len(data_list)
    for i in range(lines):
        for word in data_list[i]:
            if((word in data_dict) == False):
                cnt = len(data_dict)
                data_dict[word] = cnt
    return data_dict

def append_dict(data_list, data_dict):
    lines = len(data_list)
    for i in range(lines):
        for word in data_list[i]:
            if((word in data_dict) == False):
                cnt = len(data_dict)
                data_dict[word] = cnt
    return data_dict

def one_hot(data_list, data_dict):
    lines = len(data_list)
    total = len(data_dict)
    oh_mat = np.zeros(total*lines).reshape((lines, total))
    for i in range(lines):
        for word in data_list[i]:
            if(word in data_dict):
                oh_mat[i][data_dict[word]] = 1
    return oh_mat

def word_count(data_list):
    count_dict = {}
    if(isinstance(data_list[0],str)):
        for word in data_list:
            if((word in count_dict) == False):
                count_dict[word] = 1
            else:
                count_dict[word] = count_dict[word] + 1
    else:
        lines = len(data_list)
        for i in range(lines):
            for word in data_list[i]:
                if((word in count_dict) == False):
                    count_dict[word] = 1
                else:
                    count_dict[word] = count_dict[word] + 1
    return count_dict

def tf(data_list, data_dict):
    lines = len(data_list)
    total = len(data_dict)
    tf_mat = np.zeros(total*lines).reshape((lines, total))
    for i in range(lines):
        count_line = word_count(data_list[i])
        line_len = 0
        for word in data_list[i]:
            line_len += 1
            if(word in data_dict):
                tf_mat[i][data_dict[word]] = count_line[word]
                
        if(line_len!=0):
            tf_mat[i] /= line_len
    return tf_mat

def idf(data_list, data_dict):
    lines = len(data_list)
    total = len(data_dict)
    idf_vec = np.zeros(total)
    prev_line = np.ones(total)*-1

    for i in range(lines):
        for word in data_list[i]:
            if(word in data_dict):
                ind = data_dict[word]
                if(i!=prev_line[ind]):
                    idf_vec[ind] = idf_vec[ind] + 1
                    prev_line[ind] = i
    idf_vec = np.log(lines/(idf_vec+1))
    idf_vec[np.where(idf_vec<0)]=0
    return idf_vec

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

def classify():
    ds_path = '.\\'
    train_file = 'train_set.csv'
    train_data = pd.read_csv(ds_path+train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(ds_path+valid_file)

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
    write_path = '.\\result\\'
    cl_result = 'classification.csv'
    with open(write_path+cl_result, 'wb') as f:
        np.savetxt(f, pre, delimiter = ',', fmt = '%.6f')    

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

def regression():
    ds_path = 'C:\\study\\semaster3_1\\AI\\lab2\\DATA\\regression_dataset\\'
    train_file = 'train_set.csv'
    train_data = pd.read_csv(ds_path+train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(ds_path+valid_file)

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

    write_path = 'C:\\study\\semaster3_1\\AI\\lab2\\result\\'
    valid_result = 'valid_result.csv'
    with open(write_path+valid_result, 'wb') as f:
        np.savetxt(f, result, delimiter = ',', fmt = '%.6f')


def main():
    classify()
    # regression()
    

if __name__ == '__main__':
    main()