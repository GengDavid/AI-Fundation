#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace std;

struct Smat{
	int rows, cols, items;
	double** mat;
	Smat():rows(0),cols(0),items(0),mat(NULL){}
	Smat(int rows, int cols, int items, double ** mat):rows(rows),cols(cols),items(items),mat(mat){}
	~Smat(){for(int i=0;i<items;i++) delete[] mat[i]; delete[] mat;}
};

struct Dis{
	double distance;
	int index;
};
bool dis_cmp(const Dis& d1, const Dis& d2){
	return d1.distance<d2.distance;
}

struct Lb_cnt{
	double cnt;
	int index;
};
bool lb_cmp(const Lb_cnt& d1, const Lb_cnt& d2){
	return d1.cnt>d2.cnt;
}

Smat load_file(string file_name){
	ifstream infile; 
    infile.open(file_name.c_str());   
    string s;
    int rows, cols, items;
    int cnt = 0;
	double** smat;
    while(getline(infile,s)){
    	if(cnt==0) rows = atoi(s.c_str());
    	else if(cnt==1) cols = atoi(s.c_str());
		else if(cnt==2) {
			items = atoi(s.c_str());
			smat = new double*[items];
		}
		else{
			smat[cnt-3] = new double[3]; 
			istringstream is(s);	
			is>>smat[cnt-3][0]>>smat[cnt-3][1]>>smat[cnt-3][2];
		}
		cnt++;
    }
    infile.close();  
    return Smat(rows, cols, items, smat);
}

int labels[62522];
void load_label(string file_name){
	ifstream infile; 
    infile.open(file_name.c_str());   
    string s;
    int cnt = 0;
    while(getline(infile,s)){
    	labels[cnt++] = atoi(s.c_str());
    }
    infile.close();  
}

double L2(double x1, double x2){
	return (x1-x2)*(x1-x2);
}

int main(){
	int k = 50;
	//string ss1 = ".\\smatrix_small_train.txt";
	//string ss2 = ".\\smatrix_small_test.txt";
	//string ss3 = ".\\label_small.txt";
	string ss1 = ".\\train_tfidf.txt";
	string ss2 = ".\\test_set_smatrix.txt";
	string ss3 = ".\\transed_label.txt";	
 	Smat train_mat = load_file(ss1); //train
 	Smat test_mat = load_file(ss2); //test
 	load_label(ss3);
 	cout<<"finish reading files"<<endl;
	int cnt1 = 0, cnt2 = 0;
	int* result = new int[test_mat.rows];	
	for(int i=0;i<test_mat.rows;i++){ // for each sample
		if(i%1000==0) printf("%d ",i);
		int pre_cnt1 = cnt1;
		cnt2 = 0;
		Dis* distances = new Dis[train_mat.rows];
		for(int j=0;j<train_mat.rows;j++){  // calculate dis to train data
			double dis = 0;
			cnt1 = pre_cnt1;
			while((cnt1<test_mat.items&&cnt2<train_mat.items)&&
			(test_mat.mat[cnt1][0]==i&&train_mat.mat[cnt2][0]==j)&&
			(test_mat.mat[cnt1][1]<test_mat.cols||train_mat.mat[cnt2][1]<train_mat.cols)){
				if(test_mat.mat[cnt1][1]==train_mat.mat[cnt2][1])
					dis += L2(test_mat.mat[cnt1++][2], train_mat.mat[cnt2++][2]);
				else if(test_mat.mat[cnt1][1]<train_mat.mat[cnt2][1]) 
					dis += test_mat.mat[cnt1++][2];
				else 
					dis += train_mat.mat[cnt2++][2];			
			}
			while(cnt1<test_mat.items&&test_mat.mat[cnt1][0]==i&&test_mat.mat[cnt1][1]<test_mat.cols) 
				dis += test_mat.mat[cnt1++][2];
			while(cnt2<train_mat.items&&train_mat.mat[cnt2][0]==j&&train_mat.mat[cnt2][1]<train_mat.cols) 
				dis += train_mat.mat[cnt2++][2];		
			distances[j].distance = dis;
			distances[j].index = j;			
		}
		sort(distances,distances+train_mat.rows, dis_cmp);
		Lb_cnt* label_cnt = new Lb_cnt[3];
		for(int kk=0;kk<k;kk++){
			int d = distances[kk].distance;
			if(d==0) d = 0xffff;
			label_cnt[labels[distances[kk].index]].cnt+=1.0/d;
			label_cnt[labels[distances[kk].index]].index = labels[distances[kk].index];		
		}
		sort(label_cnt,label_cnt+3, lb_cmp);
		result[i]=label_cnt[0].index;
		delete[] label_cnt;
		delete[] distances;
	}
	string k_str, result_out = "result";
	stringstream ss;
    ss<<k;
    ss>>k_str;	
	result_out += k_str;
	result_out += ".csv";
	fstream result_file(result_out.c_str(), ios::out);
	for(int i=0;i<test_mat.rows;i++) {
		if(result[i]==0)result_file<<"LOW"<<endl;
		else if(result[i]==1)result_file<<"MID"<<endl;
		else if(result[i]==2)result_file<<"HIG"<<endl;
	}
	result_file.close();
	delete[] result;
	return 0;
}
