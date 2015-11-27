#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/matrix.h"
#include "../include/elm.h"
#define TESTDATA 10000

int main(int argc,char **argv){

	int i,j,k = 0;
	double MSE = 0,TrainingAccuracy;
	float **input,**weight,*biase,**tranpI,**tempH,**H;
	float **tranpw;
	float **train_set;
	float **T,**Y;
	float **out;
	
	train_set = (float **)calloc(TESTDATA,sizeof(float *));
	tranpI = (float **)calloc(INPUT_NEURONS,sizeof(float *));
	input = (float **)calloc(TESTDATA,sizeof(float *));        		/*datasize * INPUT_NEURONS*/
	weight = (float **)calloc(HIDDEN_NEURONS,sizeof(float *));		/*HIDDEN_NEURONS * INPUT_NEURONS*/
	biase = (float *)calloc(HIDDEN_NEURONS,sizeof(float)); 			/*HIDDEN_NEURONS*/
	tempH = (float **)calloc(HIDDEN_NEURONS,sizeof(float *)); 		/*HIDDEN_NEURONS * datasize*/
	tranpw = (float **)calloc(INPUT_NEURONS,sizeof(float *));
	H = (float **)calloc(TESTDATA,sizeof(float *));
	T = (float **)calloc(TESTDATA,sizeof(float *));
	Y = (float **)calloc(TESTDATA,sizeof(float *));
	out = (float **)calloc(HIDDEN_NEURONS,sizeof(float *));
	
	for(i=0;i<TESTDATA;i++){
		train_set[i] = (float *)calloc(NUMROWS,sizeof(float));
		input[i] = (float *)calloc(INPUT_NEURONS,sizeof(float));
		H[i] = (float *)calloc(HIDDEN_NEURONS,sizeof(float));
	}
	for(i=0;i<TESTDATA;i++)
	{
		T[i] = (float *)calloc(OUTPUT_NEURONS,sizeof(float));
		Y[i] = (float *)calloc(OUTPUT_NEURONS,sizeof(float));
	}
	for(i=0;i<INPUT_NEURONS;i++){
		tranpI[i] = (float *)calloc(TESTDATA,sizeof(float));
		tranpw[i] = (float *)calloc(HIDDEN_NEURONS,sizeof(float));
	}
	
	for(i=0;i<HIDDEN_NEURONS;i++)
	{
		weight[i] = (float *)calloc(INPUT_NEURONS,sizeof(float));
		tempH[i] = (float *)calloc(TESTDATA,sizeof(float));
		out[i] = (float *)calloc(OUTPUT_NEURONS,sizeof(float));
	}
	
	printf("begin to random weight and biase...\n");
	/*得到随机的偏置和权重*/
	//RandomWeight(weight,HIDDEN_NEURONS,INPUT_NEURONS);
	//RandomBiase(biase,HIDDEN_NEURONS);
	if(LoadMatrix(tranpw,"../result/weight",INPUT_NEURONS,HIDDEN_NEURONS,0) == 0){
		printf("load input file error!!!\n");	
		return 0;
	}
	TranspositionMatrix(tranpw,weight,INPUT_NEURONS,HIDDEN_NEURONS);
	if(LoadMatrix_s(biase,"../result/bias",1,HIDDEN_NEURONS,0) == 0){
		printf("load input file error!!!\n");	
		return 0;
	}
	/*加载数据集到内存*/
	printf("begin to load input from the file...\n");
	if(LoadMatrix(train_set,"../sample/1",TESTDATA,NUMROWS,1) == 0){
		printf("load input file error!!!\n");	
		return 0;
	}
	
	/*将数据集划分成输入和输出*/
	for(i = 0;i < TESTDATA ;i++)
	{
		T[k++][0] = train_set[i][0];
		for(j = 1;j <= INPUT_NEURONS;j++)
		{
			input[i][j-1] = train_set[i][j];
		}
	}
	/*ELM*/
	printf("begin to compute...\n");
	
	TranspositionMatrix(input,tranpI,TESTDATA,INPUT_NEURONS);
	printf("begin to compute step 1...\n");
	MultiplyMatrix_cblas(weight,HIDDEN_NEURONS,INPUT_NEURONS, tranpI,INPUT_NEURONS,TESTDATA,tempH);
	printf("begin to compute setp 2...\n");
	AddMatrix_bais(tempH,biase,HIDDEN_NEURONS,TESTDATA);
	printf("begin to compute step 3...\n");
	SigmoidHandle(tempH,HIDDEN_NEURONS,TESTDATA);
	printf("begin to compute step 4...\n");
	TranspositionMatrix(tempH,H,HIDDEN_NEURONS,TESTDATA);
	printf("begin to load input from the file...\n");
	if(LoadMatrix(out,"../result/result",HIDDEN_NEURONS,OUTPUT_NEURONS,0) == 0){
		printf("load input file error!!!\n");	
		return 0;
	}
	//检测准确率
	MultiplyMatrix_cblas(H,TESTDATA,HIDDEN_NEURONS,out,HIDDEN_NEURONS,OUTPUT_NEURONS,Y);
	for(i = 0;i< TESTDATA;i++)
	{
		MSE += (Y[i][0] - T[i][0])*(Y[i][0] - T[i][0]);
	}
	TrainingAccuracy = sqrt(MSE/TESTDATA);
	
	printf("trainning accuracy :%f\n",TrainingAccuracy);
	
	printf("test complete...\n");
	//print(PIMatrix,TESTDATA,HIDDEN_NEURONS);
		
	return 0;
}
