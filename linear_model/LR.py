import numpy as np
import pandas as pd
import time
from evaluate import evaluation
from cross_validation import *

def sigmoid(x):
	return 1/(1+np.exp(-x))

def to_lable(x):
	if(x>=0.5):
		return 1
	else:
		return 0

def LR_train(data, label, alpha, valid_data, gt, mode='SGD'):
	# data第一列全1，增广向量
	eva = np.zeros((1000,4))
	eva2 = np.zeros((1000,4))
	cnt = 0
	penalty_lambda = 10
	w = np.zeros(data.shape[1])
	err = data.shape[0]
	cnt = 0
	if(mode=='SGD'):
		for i in range(1000):
			if(i>800):
				alpha *= 0.95
			sum_g = np.zeros(data.shape[1])
			for x,y in zip(data,label):
				err = y - sigmoid(np.dot(x,w))
				w += alpha*(err*x+2*penalty_lambda*2)

			mult_result = np.sum(data*w, axis=1)
			result = np.array(list(map(to_lable,mult_result))).astype(np.int)
			eva[i,:] = evaluation(label, result)
		
	elif(mode=='GD'):
		for i in range(1000):
			if(i>800):
				alpha *= 0.95
			err = label - sigmoid(np.sum(data*w,axis=1))
			# since we use label - h, here use add
			w += alpha*(np.sum(data.transpose()*err,axis=1)+2*penalty_lambda*w)

			mult_result = np.sum(data*w, axis=1)
			result = np.array(list(map(to_lable,mult_result))).astype(np.int)
			eva[i,:] = evaluation(label, result)
			result = LR_classification(w, valid_data, gt)
			eva2[i,:] = evaluation(gt, result)

	np.savetxt('.//training_eva.csv',eva,delimiter =',',fmt='%.6f')
	np.savetxt('.//training_eva2.csv',eva2,delimiter =',',fmt='%.6f')
	return w

def LR_classification(trained_w, valid_data, gt):
	mult_result = np.sum(valid_data*trained_w, axis=1)
	result = np.array(list(map(to_lable,mult_result))).astype(np.int)
	np.savetxt('.//result.csv',result,delimiter =',',fmt='%d')
	return result

def LR(train_file, test_file, valid_flag, test_flag, mode):
	k = 5
	alpha = 0.000001
	data = pd.read_csv('.\\train.csv',header=None).values
	splited_data = split_dataset(data, k)
	cnt = 1
	eva_index = np.zeros((10,4))
	for it in range(k):	
		print('Iteration '+str(cnt))
		cnt+=1	
		if(k!=1):
			train_set = np.delete(splited_data,it,axis=0).reshape((data.shape[0]//k)*(k-1),data.shape[1])
			lable = train_set[:,-1]
			train_set = train_set[:,0:-1]
			aug_data = np.column_stack((np.ones(train_set.shape[0]).transpose(),train_set))
			valid_set = splited_data[it] # use the rest fold data as validation set
			gt = valid_set[:,-1]
			valid_set = valid_set[:,0:-1]
			augvalid_data = np.column_stack((np.ones(valid_set.shape[0]).transpose(),valid_set))
		else:
			lable = data[:,-1]
			train_set = data[:,0:-1]
			aug_data = np.column_stack((np.ones(train_set.shape[0]).transpose(),train_set))
		print('Finish reading files')
		print('begin training with '+mode)
		time_start = time.time()
		trained_w = LR_train(aug_data, lable, alpha, augvalid_data, gt, mode=mode)
		np.savetxt('.//w.csv',trained_w,delimiter =',',fmt='%d')
		time_finish= time.time()
		print('training time:'+str(time_finish-time_start)+'s')		
		if(k!=1):
			print('err on training set')
			result = LR_classification(trained_w, aug_data, lable)
			eva_index[it,:] = evaluation(lable, result)
			print(eva_index[it])	
			print('evaluate on validation set')
			result = LR_classification(trained_w, augvalid_data, gt)			
			eva_index[it,:] = evaluation(gt, result)
			print(eva_index[it])	

if __name__ == '__main__':
	mode = ['GD','SGD']
	LR('.\\train.csv','\\test.csv',1,0,mode[0])

	
	