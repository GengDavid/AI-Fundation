import numpy as np
import pandas as pd
import time

def evaluation(gt, result):
	dif = gt+2*result
	tp = np.sum(dif==3)
	tn = np.sum(dif==-3)
	fn = np.sum(dif==-1)
	fp = np.sum(dif==1)
	acc = (tp+tn)/(tp+fp+tn+fn)
	rec = tp/(tp+fn) if (tp+fn)!=0 else 0
	pre = tp/(tp+fp) if (tp+fp)!=0 else 0
	if(tp!=0):
		f1 = 2*pre*rec/(pre+rec)
	else:
		f1 = 0
	return acc,rec,pre,f1

def pla_train(data, lable, alpha, mode='SGD'):
	# data第一列全1，增广向量
	w = np.zeros(data.shape[1])
	err = data.shape[0]
	cnt = 0
	if(mode=='GD'):
		while(cnt<2000):
			sum_g = np.zeros(data.shape[1])
			for x,y in zip(data,lable):
				mult = -np.dot(x,w)*y
				if(mult>=0):
					sum_g += alpha*x*y
			w += sum_g
			cnt += 1
	elif(mode=='SGD'):
		while(cnt<1000):
			for x,y in zip(data,lable):
				mult = -np.dot(x,w)*y
				if(mult>=0):
					w += alpha*x*y
			cnt += 1
	elif(mode=='pocket'):
		pocket_w = w
		pocket_score = (0,0,0,0)
		temp = -(np.sum(data*w,axis=1)*lable)
		temp_result = np.array(list(map(sign,np.sum(data*w, axis=1)))).astype(np.int)
		# store misclassified point index
		wrong = np.where(temp>0)[0]	
		if(len(wrong)!=0):
			err = wrong[0]
		else:
			err = 0
		while(cnt<500000):
			eval_res = evaluation(lable, temp_result)
			# print(eval_res)
			if(eval_res[0]>pocket_score[0]):
				# update the best w
				pocket_w=w
				pocket_score=eval_res		
			# random select a misclassified to update w
			if(err==0):	
				i=0
			else:
				i = np.random.randint(err)
				i = wrong[i]
			w += alpha*lable[i]*data[i]
			temp = -(np.sum(data*w,axis=1)*lable)
			temp_result = np.array(list(map(sign,np.sum(data*w, axis=1)))).astype(np.int)
			wrong = np.where(temp>0)[0]
			err = wrong.shape[0]
			cnt += 1
		w = pocket_w		
	return w

def sign(x):
	return 1 if x>0 else -1

def pla_classification(trained_w, valid_data, gt):
	mult_result = np.sum(valid_data*trained_w, axis=1)
	result = np.array(list(map(sign,mult_result))).astype(np.int)
	np.savetxt('.//result.csv',result,delimiter =',',fmt='%d')
	acc,rec,pre,f1 = evaluation(gt,result)
	print("accuracy", acc)
	print("recall", rec)
	print("precision", pre)
	print("f1", f1)

def pla(train_file, valid_file, test_file, valid_flag, test_flag, mode):
	data = pd.read_csv(train_file, header=None).values
	lable = data[:,-1]
	data = data[:,0:-1]
	aug_data = np.column_stack((np.ones(data.shape[0]).transpose(),data))
	print('Finish reading files')
	print('begin training with,'+mode)
	time_start = time.time()
	trained_w = pd.read_csv('.\\w.csv',header=None).values
	print(trained_w)
	np.savetxt('.//w.csv',trained_w,delimiter =',',fmt='%d')
	time_finish= time.time()
	print('training time:'+str(time_finish-time_start)+'s')
	valid_data = pd.read_csv(valid_file, header=None).values
	gt = valid_data[:,-1]
	valid_data = valid_data[:,0:-1]
	augvalid_data = np.column_stack((np.ones(valid_data.shape[0]).transpose(),valid_data))
	print('evaluate on validation set')
	pla_classification(trained_w, augvalid_data, gt)


if __name__ == '__main__':
	mode = ['GD','SGD','pocket']
	pla('.\\train.csv','.\\test.csv','.\\test.csv',1,0,mode[2])

	
	