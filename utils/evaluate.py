import numpy as np

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