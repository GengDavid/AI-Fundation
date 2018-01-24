from dt import *
from random import randrange
from evaluate import evaluation 

def get_sample(train_data, sample_ratio, feature_number):
	sample_size = round(train_data.shape[0]*sample_ratio)
	sample_data = []
	exfit_sample_data = np.array([])
	used_data = np.zeros(train_data.shape[0])
	while (len(sample_data) < sample_size):
		# sampling with replacement
		index = randrange(train_data.shape[0])
		sample_data.append(train_data[index])
		used_data[index]=1
	sample_data = np.array(sample_data)
	for feature in feature_number:
		if(exfit_sample_data.shape[0]==0):
			exfit_sample_data = np.append(exfit_sample_data, sample_data[:,feature])
		else:
			exfit_sample_data = np.column_stack((exfit_sample_data, sample_data[:,feature]))
	exfit_sample_data = np.column_stack((exfit_sample_data, sample_data[:,-1]))
	res_index = np.where(used_data==0)[0]
	res_data = train_data[res_index]
	return exfit_sample_data,res_data

def random_forest(train_data, sample_ratio, n_trees, n_features):
	forest = []
	for i in range(n_trees):
		feature_number = []
		f = 0
		while(f<n_features):
			rd_feature = randrange(len(train_data[1])-1)
			if rd_feature not in feature_number:
				feature_number.append(rd_feature)
				f += 1 
		# store teh original column index of each column
		feature_number = np.array(feature_number)
		sample_data,res_data = get_sample(train_data, sample_ratio, feature_number)
		# print('data prepared')
		tree = dt_train(sample_data, feature_number, 'ID3')
		forest.append(tree)
		tree = dt_train(sample_data, feature_number, 'C4.5')
		forest.append(tree)
		#tree = dt_train(sample_data, feature_number, 'CART')
		#forest.append(tree)
	return forest,res_data

def rf_classification(forest, test_data, validation = 0):
	result_lables = []
	majority_lable = []
	if(validation==1):
		sub_val_result = np.zeros((test_data.shape[0],4))
		cnt = 0
	for tree in forest:
		res = dt_classification(tree, test_data)
		result_lables.append(res)
		if(validation == 1):
			sub_val_result[cnt,:]=evaluation(test_data[:,-1], res)
			cnt+=1
	result_lables = np.array(result_lables)
	for i in range(result_lables.shape[1]):
		result_counter = Counter(result_lables[:,i])
		# print(result_counter)
		majority_lable.append(result_counter.most_common()[0][0])
	if(validation==1):
		return np.array(majority_lable),np.array(sub_val_result)
	return np.array(majority_lable)

if __name__ == '__main__':
	# data = np.array([[1,1,1],[1,1,1],[1,0,-1],[0,1,-1],[0,1,-1]])
	data = pd.read_csv('.\\train.csv',header=None).values
	sample_ratio = 0.9
	n_trees = 55
	#n_features = int(np.log2(data.shape[1]))
	n_features = int(data.shape[1]*0.75)
	rd_forest,valid_set = random_forest(data, sample_ratio, n_trees, n_features)
	#print(rd_forest)
	#result, sub_val_result = rf_classification(rd_forest, valid_set, 1)
	#np.savetxt('.//evaluation_for_each_tree.csv',sub_val_result,delimiter =',',fmt='%.6f')
	result = rf_classification(rd_forest, valid_set)
	print(evaluation(valid_set[:,-1], result))
	test_set = pd.read_csv('.\\test.csv',header=None).values[:,:-1]
	test_result = rf_classification(rd_forest, test_set)
	np.savetxt('.//15352403_zhanggengwei.txt',test_result,fmt='%d')