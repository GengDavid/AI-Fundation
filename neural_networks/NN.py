import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from evaluate import evaluation
from cross_validation import *

def tanh(x):
	return np.tanh(x)

def d_tanh(x):
	return 1 - x**2

def sigmoid(x):
	return 1/(1+np.exp(-x))

def d_sigmoid(x):
	'''
	input x is the output of a layer
			it's the result of sigmoid
			so no need to calculate sigmoid in this function
	'''
	return x*(1-x)

def ReLU(x):
	ret = np.array(x)
	ret[ret<0] = 0
	return ret

def d_ReLU(x):
	ret = np.array(x)
	ret[ret>0] = 1
	return ret

def linear(x):
	return 0.01*x

def d_linear(x):
	return 0.01

class HidenLayer:
	def __init__(self, lr, numInput, numOutput, activation = 'tanh', fromModel = 0):
		self.lr = lr
		self.numOutput = numOutput
		self.numInput = numInput
		if(fromModel==0):
			self.w = np.random.rand(numOutput, numInput)
		else:
			self.w = fromModel
		self.output = np.zeros(numOutput)
		self.input = np.zeros(numInput)
		if(activation == 'tanh'):
			self.activation = tanh
			self.d_activation = d_tanh
		elif(activation == 'sigmoid'):
			self.activation = sigmoid
			self.d_activation = d_sigmoid
		elif(activation == 'ReLU'):
			self.activation = ReLU
			self.d_activation = d_ReLU
		elif(activation == 'linear'):
			self.activation = linear
			self.d_activation = d_linear

	def forward(self, input_x):
		self.input = input_x
		self.output = self.activation(np.dot(self.w, input_x))

	def backward(self, error, output_layer = 0):
		new_err = np.zeros(self.numInput)
		error *= self.d_activation(self.output)
		for k, e in enumerate(error):
			new_err += e*self.w[k]
			self.w[k] -= self.lr*e*self.input	
		# the bias has no input
		return np.delete(new_err, 0)



class MyNN:
	def __init__(self, data, label, valid_data, gt, epoch, batch_size = 1):
		self.data = data
		self.label = label
		self.valid_data = valid_data
		self.gt = gt
		self.epoch = epoch
		self.layer = []
		self.batch_size = batch_size

	def fit(self, x, index):
		# forward propagation
		self.layer[0].forward(x)
		self.layer[1].forward(np.hstack((1,self.layer[0].output)))

		
		# derivation of the output layer loss
		output_err = -(self.label[index] - self.layer[1].output) 
		e2 = output_err**2
		# back propagation
		#print(self.layer[1].w)
		errs = self.layer[1].backward(output_err, output_layer = 1)
		#print(self.layer[0].w)	
		errs = self.layer[0].backward(errs)	
		return e2,self.layer[1].output

	def train(self):
		num_data = self.data.shape[0]
		num_feature = self.data.shape[1] if len(self.data.shape)>1 else self.data.shape[0]
		# defination of the network
		self.layer.append(HidenLayer(0.001,num_feature,num_feature))
		self.layer.append(HidenLayer(0.001,self.layer[0].output.shape[0]+1,1,activation = 'linear'))
		#print(self.layer[1].w)
		save_training_err = np.zeros(self.epoch)
		save_valid_err = np.zeros(self.epoch)
		for epc in range(self.epoch):
			if(len(self.data.shape)>1):
				ee = 0
				for index, x in enumerate(self.data):
					e, o = self.fit(x, index)
					ee += e
				if(epc%50==0):
					print('epoch'+str(epc))
					print(ee/self.data.shape[0])
					self.save_model('.\\trained_model\\trained_model_iteration_'+str(epc)+'.mat')
				save_training_err[epc] = 0.5*ee/self.data.shape[0]
				valid_output = nn.test(self.valid_data)
				save_valid_err[epc] = 0.5*np.sum(np.square(self.gt-valid_output.transpose()))/valid_output.shape[0]

				
			else:
				x = self.data
				self.fit(x, 0)
		np.savetxt('.\\save_training_err.csv', save_training_err, delimiter = ',', fmt = "%.4f")
		np.savetxt('.\\save_valid_err.csv', save_valid_err, delimiter = ',', fmt = "%.4f")


	def test_sample(self, x):
		self.layer[0].forward(x)
		self.layer[1].forward(np.hstack((1,self.layer[0].output)))
		
		return self.layer[1].output

	def test(self, data):
		output = []
		for x in data:
			output.append(self.test_sample(x))
		return np.array(output)

	def save_model(self, save_file):
		model_dic = {}
		cnt = 0
		for l in self.layer:
			model_dic['layer'+str(cnt)] = l.w
			cnt += 1
		sio.savemat(save_file, model_dic) 

	def load_trained_model(self):
		load_file = '.\\final.mat'
		load_model = sio.loadmat(load_file)
		cnt = 0
		while('layer'+str(cnt) in load_model):
			model_w = load_model['layer'+str(cnt)]
			act = 'tanh' if cnt<2 else 'linear'
			self.layer.append(HidenLayer(0.001,model_w.shape[1],model_w.shape[0], activation = act, fromModel=model_w))
			cnt+=1		



if __name__ == '__main__':
	#data = np.array([1,0.05,0.1])
	#label = np.array([0.01,0.99])
	data = pd.read_csv('.\\transformed_train.csv', header = None).values
	data = data[1:,:]

	
	k = 10
	splited_data = split_dataset(data, k)
	for it in range(k):	
		if(k!=1):
			train_set = np.delete(splited_data,it,axis=0).reshape((data.shape[0]//k)*(k-1),data.shape[1])
			label = train_set[:,-1]
			train_set = train_set[:,0:-1]
			aug_data = np.column_stack((np.ones(train_set.shape[0]).transpose(),train_set))
			valid_set = splited_data[it] # use the rest fold data as validation set
			gt = valid_set[:,-1]
			valid_set = valid_set[:,0:-1]
			augvalid_data = np.column_stack((np.ones(valid_set.shape[0]).transpose(),valid_set))
		else:
			label = data[:,-1]
			train_set = data[:,0:-1]
			aug_data = np.column_stack((np.ones(train_set.shape[0]).transpose(),train_set))

		nn = MyNN(aug_data, label, augvalid_data, gt, 10000)
		nn.train()
		nn.save_model(".\\trained_model\\final.mat")

		if(k!=1):
			# nn.load_trained_model()

			# output = nn.test(aug_data)
			# np.savetxt('.\\pred_result_on_valid.csv', output, delimiter = ',', fmt = "%d")
			# np.savetxt('.\\gt.csv', label, delimiter = ',', fmt = "%d")	
			# print('err on training set')
			# mse = 0.5*np.sum(np.square(label-output.transpose()))/output.shape[0]
			# print('training mse: '+str(mse))
		
			print('evaluate on validation set')
			output = nn.test(augvalid_data)
			#np.savetxt('.\\pred_result_on_valid.csv', output, delimiter = ',', fmt = "%d")
			#np.savetxt('.\\gt.csv', gt, delimiter = ',', fmt = "%d")	
			mse = 0.5*np.sum(np.square(gt-output.transpose()))/output.shape[0]
			print('validation mse: '+str(mse))
			break	

