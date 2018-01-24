import numpy as np
import operator
import pandas as pd
import math

def get_word_list(data_list):
    data_dict = {}
    word_list = []
    lines = len(data_list)
    for i in range(lines):
        for word in data_list[i]:
            if((word in data_dict) == False):
                word_list.append(word)
                data_dict[word] = 1
    return word_list


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
    cnt = 0
    if(isinstance(data_list[0],str)):
        for word in data_list:
            if((word in count_dict) == False):
                count_dict[word] = 1
            else:
                count_dict[word] = count_dict[word] + 1
            cnt += 1
    else:
        lines = len(data_list)
        for i in range(lines):
            for word in data_list[i]:
                if((word in count_dict) == False):
                    count_dict[word] = 1
                else:
                    count_dict[word] = count_dict[word] + 1
                cnt+=1
    return count_dict,cnt

def tf(data_list, data_dict):
    lines = len(data_list)
    total = len(data_dict)
    tf_mat = np.zeros(total*lines).reshape((lines, total))
    line_len = np.zeros(lines)
    for i in range(lines):
        count_line, cnt = word_count(data_list[i])
        for word in data_list[i]:
            if(word in data_dict):
                tf_mat[i][data_dict[word]] = count_line[word]
                line_len[i] += 1
    return tf_mat, line_len

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
    return idf_vec

def nb_clssify_train(train_data, train_lables, data_dict, mode):
    emotion_list = ['anger','disgust','fear','joy','sad','surprise']
    oh_mat = one_hot(train_data,data_dict)
    total = train_lables.shape[0]
    features = oh_mat.shape[1]
    prior = np.zeros(6)
    filter = np.zeros((6,total)).astype(bool)
    likelihood = np.zeros((6,features))
    word_list = get_word_list(train_data)

    for i in range(6):
        filter[i] = train_lables==emotion_list[i]
        prior[i] = np.sum(filter[i])

    for i in range(6):
        if(mode=='bernoulli'):
            data_with_em = oh_mat[filter[i].transpose()]
            for j in range(features):
                likelihood[i][j] = (np.sum(data_with_em[:,j])+1)/(prior[i]+features)
        elif(mode=='multinomial'):
            data_with_em = train_data[filter[i].transpose()]
            word_in_bag,cnt = word_count(data_with_em)            
            for j in range(len(word_list)):
                if(word_list[j] in word_in_bag):
                    likelihood[i][j] = math.log((word_in_bag[word_list[j]]+1)/(cnt+features))
                else:
                    likelihood[i][j] = math.log(1/(cnt+features))
    prior = (prior+1)/(total+6)
    return np.log(prior), likelihood

def nb_clssify(prior, likelihood, data):
    emotion_list = ['anger','disgust','fear','joy','sad','surprise']
    numbers = data.shape[0]
    result = np.zeros((numbers,6))
    result_lable = np.zeros(numbers).astype(str)
    for i in range(numbers):
        sample = np.array(np.nonzero(data[i])) # a tuple, sample[0]is the index of nonzeor item
        max_prob = -0xffff
        max_lb = 0
        for j in range(6):
            result[i][j] = prior[j]
            for k in range(sample.shape[1]):
                result[i][j] += likelihood[j][sample[0][k]]
            if(result[i][j]>max_prob):
                max_prob=result[i][j]
                max_lb = j     
        result_lable[i] = emotion_list[max_lb]
    return result_lable
    
def classify():
    ds_path = '.\\'
    train_file = 'train_set.csv'
    train_data = pd.read_csv(ds_path+train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(ds_path+valid_file)

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

    write_path = '.\\result\\'

def nb_regression_train(tf_mat, line_len, train_lables, alpha = 0.001):
    tnum = tf_mat.shape[0]
    features = tf_mat.shape[1]
    emotions = train_lables.shape[1]  
    likelihood = np.zeros((tnum,features))  
    for k in range(tnum):
        likelihood[k] = (tf_mat[k]+alpha)/(line_len[k]+alpha*features)
    return likelihood

def nb_regression(likelihood, valid_oh, train_lables):
    tnum = likelihood.shape[0]
    vnum = valid_oh.shape[0]
    emotions = train_lables.shape[1]
    result = np.zeros((vnum,emotions))

    valid_oh = valid_oh.astype(bool)
    # for each validation sample (i)
    # calculate probability of each emotion (j)
    for i in range(vnum):
        for j in range(emotions):
            for k in range(tnum):
                if(train_lables[k][j]!=0):
                    prob = train_lables[k][j]
                else:
                    continue
                vlikelihood=likelihood[k][valid_oh[i]]
                for x in range(vlikelihood.shape[0]):
                    prob *= vlikelihood[x]
                result[i][j]+=prob
    

    for i in range(vnum):
        result[i] = result[i]/np.sum(result[i])

    return result

def regression():
    ds_path = '.\\'
    train_file = 'train_set.csv'
    train_data = pd.read_csv(ds_path+train_file)

    valid_file = 'validation_set.csv'
    valid_data = pd.read_csv(ds_path+valid_file)

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

    write_path = '.\\result\\'
    valid_result = 'nb_valid_result.csv'
    with open(write_path+valid_result, 'wb') as f:
        np.savetxt(f, result, delimiter = ',', fmt = '%.4f')


def main():
    classify()
    # regression()
    

if __name__ == '__main__':
    main()