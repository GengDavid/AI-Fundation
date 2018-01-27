import numpy as np

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