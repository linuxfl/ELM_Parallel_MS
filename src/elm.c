#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "../include/elm.h"
#include "../include/matrix.h"

void Init_random()
{
	srand(time(0));
}

float RandomFloat(float start,float end)
{
	return start + (end - start) * rand() / (RAND_MAX + 1.0);
}

void print(float **matrix,int row,int column)
{
	int i,j;
	for(i=0;i<row;i++)
	{
		for(j=0;j<column;j++)
		{
			printf("%f ",matrix[i][j]);
		}
		printf("\n");
	}	
}

void RandomWeight(float **weight,int row,int column)
{
	int i,j;
	for(i=0;i<row;i++)
	{
		for(j=0;j<column;j++)
		{
			weight[i][j] = RandomFloat(-0.5,0.5);
		}
	}
}
void RandomWeight_s(float *weight,int row,int column){
	int i;
	for(i=0;i<row*column;i++)
	{
		weight[i] = RandomFloat(-0.5,0.5);
	}
}
void RandomBiase(float *biase,int row)
{
	int i;
	for(i=0;i<row;i++)
	{
		biase[i] = RandomFloat(-0.5,0.5);
	}
}

void SigmoidHandle(float **matrix,int row,int column)
{
	int i,j;
	float temp;
	for(i=0;i<row;i++){
		for(j=0;j<column;j++)
		{
			temp = matrix[i][j];
			matrix[i][j] = 1/(1+exp(-(temp)));
		}
	}
}
void SigmoidHandle_s(float *matrix,int row,int column)
{
	int i;
	float temp;
	for(i=0;i<row*column;i++){
		temp = matrix[i];
		matrix[i] = 1/(1+exp(-(temp)));
	}
}
//回归
void ELMTrain()
{
	double starttime,endtime;

	int i,j,k = 0;
	double MSE = 0,TrainingAccuracy;
	float **input,**weight,*biase,**tranpI,**tempH,**H;
	float **PIMatrix;
	float **train_set;
	float **T,**Y;
	float **out;
	
	train_set = (float **)calloc(DATASET,sizeof(float *));
	tranpI = (float **)calloc(INPUT_NEURONS,sizeof(float *));
	input = (float **)calloc(DATASET,sizeof(float *));        		/*datasize * INPUT_NEURONS*/
	weight = (float **)calloc(HIDDEN_NEURONS,sizeof(float *));		/*HIDDEN_NEURONS * INPUT_NEURONS*/
	biase = (float *)calloc(HIDDEN_NEURONS,sizeof(float)); 			/*HIDDEN_NEURONS*/
	tempH = (float **)calloc(HIDDEN_NEURONS,sizeof(float *)); 		/*HIDDEN_NEURONS * datasize*/
	PIMatrix = (float **)calloc(HIDDEN_NEURONS,sizeof(float *));
	H = (float **)calloc(DATASET,sizeof(float *));
	T = (float **)calloc(DATASET,sizeof(float *));
	Y = (float **)calloc(DATASET,sizeof(float *));
	out = (float **)calloc(HIDDEN_NEURONS,sizeof(float *));
	
	for(i=0;i<DATASET;i++){
		train_set[i] = (float *)calloc(NUMROWS,sizeof(float));
		input[i] = (float *)calloc(INPUT_NEURONS,sizeof(float));
		H[i] = (float *)calloc(HIDDEN_NEURONS,sizeof(float));
	}
	for(i=0;i<DATASET;i++)
	{
		T[i] = (float *)calloc(OUTPUT_NEURONS,sizeof(float));
		Y[i] = (float *)calloc(OUTPUT_NEURONS,sizeof(float));
	}
	for(i=0;i<INPUT_NEURONS;i++)
		tranpI[i] = (float *)calloc(DATASET,sizeof(float));
	
	for(i=0;i<HIDDEN_NEURONS;i++)
	{
		weight[i] = (float *)calloc(INPUT_NEURONS,sizeof(float));
		tempH[i] = (float *)calloc(DATASET,sizeof(float));
		out[i] = (float *)calloc(OUTPUT_NEURONS,sizeof(float));
		PIMatrix[i] = (float *)calloc(DATASET,sizeof(float));
	}
	
	printf("begin to random weight and biase...\n");
	/*得到随机的偏置和权重*/
	RandomWeight(weight,HIDDEN_NEURONS,INPUT_NEURONS);
	RandomBiase(biase,HIDDEN_NEURONS);
	
	/*加载数据集到内存*/
	printf("begin to load input from the file...\n");
	if(LoadMatrix(train_set,"../TrainingDataSet/covtype",DATASET,NUMROWS,1) == 0){
		printf("load input file error!!!\n");	
		return;
	}
	
	/*将数据集划分成输入和输出*/
	for(i = 0;i < DATASET ;i++)
	{
		T[k++][0] = train_set[i][0];
		for(j = 1;j <= INPUT_NEURONS;j++)
		{
			input[i][j-1] = train_set[i][j];
		}
	}
	SaveMatrix(input,"./result/input",DATASET,INPUT_NEURONS);
	/*ELM*/
	printf("begin to compute...\n");
	starttime = omp_get_wtime();	
	TranspositionMatrix(input,tranpI,DATASET,INPUT_NEURONS);
	printf("begin to compute step 1...\n");
	MultiplyMatrix(weight,HIDDEN_NEURONS,INPUT_NEURONS, tranpI,INPUT_NEURONS,DATASET,tempH);
	printf("begin to compute setp 2...\n");
	AddMatrix_bais(tempH,biase,HIDDEN_NEURONS,DATASET);
	printf("begin to compute step 3...\n");
	SigmoidHandle(tempH,HIDDEN_NEURONS,DATASET);
	printf("begin to compute step 4...\n");
	TranspositionMatrix(tempH,H,HIDDEN_NEURONS,DATASET);
	PseudoInverseMatrix(H,DATASET,HIDDEN_NEURONS,PIMatrix);
	MultiplyMatrix(PIMatrix,HIDDEN_NEURONS,DATASET,T,DATASET,OUTPUT_NEURONS,out);

	//SaveMatrix(H,"./result/H",DATASET,HIDDEN_NEURONS);
	//SaveMatrix(PIMatrix,"./result/PIMatrix",HIDDEN_NEURONS,DATASET);

	
	printf("begin to compute step 5...\n");
	endtime = omp_get_wtime();
	//保存输出权值
	SaveMatrix(out,"./result/result",HIDDEN_NEURONS,OUTPUT_NEURONS);
	
	//检测准确率
	MultiplyMatrix(H,DATASET,HIDDEN_NEURONS,out,HIDDEN_NEURONS,OUTPUT_NEURONS,Y);
	for(i = 0;i< DATASET;i++)
	{
		MSE += (Y[i][0] - T[i][0])*(Y[i][0] - T[i][0]);
	}
	SaveMatrix(T,"./result/t",DATASET,OUTPUT_NEURONS);
	SaveMatrix(Y,"./result/y",DATASET,OUTPUT_NEURONS);
	TrainingAccuracy = sqrt(MSE/DATASET);
	
	printf("use time :%f\n",endtime - starttime);
	printf("trainning accuracy :%f\n",TrainingAccuracy);
	
	printf("train complete...\n");
	//print(PIMatrix,DATASET,HIDDEN_NEURONS);
}
