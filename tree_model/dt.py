import numpy as np
import pandas as pd
from collections import Counter
from cross_validation import split_dataset
from visible import *
from evaluate import evaluation

first = 0

def cal_entropy(data):
	# if there is only one sample in dataset,
	# the entropy of it is 0
	if(len(data.shape)==1):
		return 0	
	data_num = data.shape[0]
	lb_cnt = {1:0,-1:0}
	for sample in data:
		lb_cnt[sample[-1]] += 1
	entropy = 0
	for key in lb_cnt:
		p = lb_cnt[key]/data_num
		if(p==0): # 0log0 = 0
			entropy = 0
		else:
			entropy -= p*np.log2(p)
	return entropy

def cal_gini(data):
	if(len(data.shape)==1):
		return 0	
	data_num = data.shape[0]
	lb_cnt = {1:0,-1:0}
	for sample in data:
		if(sample[-1]==0):
			print(data)
		lb_cnt[sample[-1]] += 1
	gini = 1
	for key in lb_cnt:
		p = lb_cnt[key]/data_num
		gini -= p**2
	return gini

def get_subdata(data, feature_value, column, get_rest = 0):
	sub = np.array([])
	rest = np.array([])
	for sample in data:
		# delete used feature
		if(sample[column]==feature_value):
			if(sub.shape[0]==0):
				sub = np.append(sub,np.delete(sample,column))
			else:
				sub = np.row_stack((sub,np.delete(sample,column)))
		else:
			if(rest.shape[0]==0):
				rest = np.append(rest,np.delete(sample,column))
			else:
				rest = np.row_stack((rest,np.delete(sample,column)))
	if(get_rest==1):
		return sub,rest
	return sub


def id3(data, feature_number):
	Hd = cal_entropy(data)
	best = 0 # the larger the better
	best_feature = 0
	# calculte information gain of each feature
	for i in range(len(feature_number)):
		feature_values = set([sample[i] for sample in data])
		condition_entropy = 0
		for feature_value in feature_values:
			subdata = get_subdata(data,feature_value,i)
			p = subdata.shape[0]/data.shape[0]
			condition_entropy += p*cal_entropy(subdata)
		inform_gain = Hd - condition_entropy
		if(inform_gain>best):
			best = inform_gain
			best_feature = i
	return best_feature

def C45(data, feature_number):
	Hd = cal_entropy(data)
	best = 0 # the larger the better
	best_feature = 0
	# calculte information gain ratio of each feature
	for i in range(len(feature_number)):
		feature_values = set([sample[i] for sample in data])
		condition_entropy = 0
		Hi = 0
		for feature_value in feature_values:
			subdata = get_subdata(data,feature_value,i)
			p = subdata.shape[0]/data.shape[0]
			sub_entropy = cal_entropy(subdata)
			# calculate condition_entropy and the entropy of a feature
			condition_entropy += p*sub_entropy
			Hi += sub_entropy
		if(Hi==0):
			inform_gain = Hd
			inform_gain_ratio = 1
		else:
			inform_gain = Hd - condition_entropy
			inform_gain_ratio = inform_gain / Hi
		if(inform_gain_ratio>best):
			best = inform_gain_ratio
			best_feature = i
	return best_feature # return the best feature's index

def cart(data, feature_number):
	best_gini = 0xff # the small the better
	best_feature = 0
	best_branch = 0
	for i in range(len(feature_number)):
		feature_values = set([sample[i] for sample in data])
		for feature_value in feature_values:
			# calculate gini index for each feature of every condition
			subdata,restdata = get_subdata(data,feature_value,i,1)
			p = subdata.shape[0]/data.shape[0]
			sub_gini = cal_gini(subdata)
			rest_gini = cal_gini(restdata)
			part_gini = p*sub_gini+(1-p)*rest_gini
			# print('gini index: ',part_gini)
			if(part_gini<best_gini):
				best_gini = part_gini
				best_feature = i
				best_branch = feature_value
	return best_feature,best_branch


def dt_train(data, feature_number, gen_algorithm):
	# print(data)
	if(len(data.shape)>1):
		label = data[:,-1]
	else:
		return data[-1]
	label_count = Counter(label)
	# if all the data have the same label,
	# return leaf
	if(len(label_count)==1):
		return label[0]
	# if it's not leaf, data has at least 2 samples
	# if there is no feature to use, return the majority class
	if(data.shape[1]==1):
		return label_count.most_common()[0][0]	
	if(feature_number.shape[0]==0):
		return label_count.most_common()[0][0]
	best_feature=0
	if(gen_algorithm=='ID3'):		
		best_feature = id3(data, feature_number)
	elif(gen_algorithm=='C4.5'):
		best_feature = C45(data, feature_number)
	elif(gen_algorithm=='CART'):
		best_feature,best_branch = cart(data, feature_number)
	# use a number(lable) to represent a feature
	bestlabel = feature_number[best_feature]
	new_feature_number=np.delete(feature_number,best_feature)
	tree = {bestlabel:{}}
	if(gen_algorithm=='CART'):
		# for CART, generate a binary tree
		tree[bestlabel][best_branch] = \
			dt_train(get_subdata(data,best_branch,best_feature),new_feature_number,gen_algorithm)
		if(get_subdata(data,best_branch,best_feature,1)[1].shape[0]!=0):
			tree[bestlabel]['others'] = \
				dt_train(get_subdata(data,best_branch,best_feature,1)[1],new_feature_number,gen_algorithm)
	else:
		feature_values = set([sample[best_feature] for sample in data])
		for feature_value in feature_values:
			tree[bestlabel][feature_value] = \
				dt_train(get_subdata(data,feature_value,best_feature),new_feature_number,gen_algorithm)
	return tree

def dt_classification_each(tree, test_data):
	node = list(tree.keys())[0]
	child = tree[node]
	for key in child.keys():
		if(test_data[node]==key):
			if(isinstance(child[key],dict)):
				return dt_classification_each(child[key],test_data)
			else:
				return child[key]
		elif(key=='others'):			
			if(isinstance(child[key],dict)):
				return dt_classification_each(child[key],test_data)
			else:
				return child[key]
	return 1

def dt_classification(tree, test_set):
	pred_result = []
	for tdata in test_set:
		res = dt_classification_each(tree,tdata)
		pred_result.append(res)
	pred_result = np.array(pred_result)
	return pred_result

if __name__ == '__main__':
	#data = np.array([[1,1,1],[1,1,1],[1,0,-1],[0,1,-1],[0,1,-1]])
	#data = pd.read_csv('.\\my_train.csv',header=None).values
	k = 10
	data = pd.read_csv('.\\train.csv',header=None).values
	featurenumber = np.arange(data.shape[1]-1)
	splited_data = split_dataset(data, k)
	cnt = 1
	eva_index = np.zeros((10,4))
	for it in range(k):	
		print('Iteration '+str(cnt))
		cnt+=1	
		if(k!=1):
			train_set = np.delete(splited_data,it,axis=0).reshape((data.shape[0]//k)*(k-1),data.shape[1])
			valid_set = splited_data[it] # use the rest fold data as validation set
		else:
			train_set = data
		tree = dt_train(train_set,featurenumber,'CART')
		if(k!=1):
			result = dt_classification(tree, valid_set)
			eva_index[it,:] = evaluation(valid_set[:,-1], result)
			print(eva_index[it])
			
	np.savetxt('.//evaluation_indicators_for_CART.csv',eva_index,delimiter =',',fmt='%.6f')

	#createPlot(tree)	
	#test_data = pd.read_csv('.\\test.csv',header=None).values
	#print(dt_classification(tree,test_data))