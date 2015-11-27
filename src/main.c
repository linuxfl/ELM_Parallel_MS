#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mpi.h"
#include "elm.h"
#include "matrix.h"
#define SAMPLE_DIR "../TrainingDataSet/mv"

int main(int argc,char **argv){
	
	int m;
	int rank,size;
	double starttime,endtime;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//将样本集均等划分，主进程不处理样本，子进程处理样本
	m = DATASET/(size);
	//printf(" m = %d\n",m);
	if(rank == 0){
		int i,j = 0,k = 0;
		float *Ht,*Hh,*tempht,*temphh,*result,*train_set,*T,*input,*tempI,*tranpH,*Y,*data_set;
		data_set = (float *)calloc(DATASET * NUMROWS,sizeof(float)); 
		train_set = (float *)calloc(m * NUMROWS,sizeof(float)); 
		T = (float *)calloc(m * OUTPUT_NEURONS,sizeof(float)); 					/* m * OUTPUT_NEURONS */
		input = (float *)calloc(m * INPUT_NEURONS,sizeof(float)); 
		tempI = (float *)calloc(m * HIDDEN_NEURONS,sizeof(float)); 
		result = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float));	/* HIDDEN_NEURONS * OUTPUT_NEURONS */
		Ht = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float)); 		/* HIDDEN_NEURONS * OUTPUT_NEURONS */
		tempht = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float)); 	/* HIDDEN_NEURONS * OUTPUT_NEURONS */
		Hh = (float *)calloc(HIDDEN_NEURONS * HIDDEN_NEURONS,sizeof(float)); 		/* HIDDEN_NEURONS * HIDDEN_NEURONS */
		temphh = (float *)calloc(HIDDEN_NEURONS * HIDDEN_NEURONS,sizeof(float)); 	/* HIDDEN_NEURONS * HIDDEN_NEURONS */
		tranpH = (float *)calloc(HIDDEN_NEURONS * m,sizeof(float)); 				/* m * HIDDEN_NEURONS */
		Y = (float *)calloc(m * OUTPUT_NEURONS,sizeof(float)); 

#if (ELM_TYPE == CLASSIFICATION_TRAINING)
		float *tempT = (float *)calloc(m,sizeof(float));
#endif
		//初始化
		/*权重在计算过程中需要进行转置，因为参数全是随机，所以为了方便计算，
		开始定义就进行转置，原行列为HIDDEN_NEURONS*INPUT_NEURONS*/
		printf("beging to train...\n");
		starttime = MPI_Wtime();
		float *weight = (float *)calloc(INPUT_NEURONS*HIDDEN_NEURONS,sizeof(float)); /*INPUT_NEURONS * HIDDEN_NEURONS */
		float *bias = (float *)calloc(HIDDEN_NEURONS,sizeof(float)); 
		//Init_random();
		RandomWeight_s(weight,INPUT_NEURONS,HIDDEN_NEURONS);
		SaveMatrix_s(weight,"./result/weight",INPUT_NEURONS,HIDDEN_NEURONS);	
		RandomBiase(bias,HIDDEN_NEURONS);
		SaveMatrix_s(bias,"./result/bias",1,HIDDEN_NEURONS);	
		//随机权重和偏置，并将权重和偏置广播给所有子进程
		//printf("file:%s,func:%s,line:%d rank:%d\n",__FILE__,__func__,__LINE__,rank);
		MPI_Bcast(weight,INPUT_NEURONS * HIDDEN_NEURONS,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(bias,HIDDEN_NEURONS,MPI_FLOAT,0,MPI_COMM_WORLD);

		//从本地读取相应的样本集
		printf("direction = %s\n",SAMPLE_DIR);
		if(LoadMatrix_s(data_set,SAMPLE_DIR,DATASET,NUMROWS,1) == 0){
			printf("rank %d:load input file error!!!\n",rank);
			MPI_Abort(MPI_COMM_WORLD,-1);
		}
		
		//scatter the dataset to slave
		MPI_Scatter(data_set,m * NUMROWS,MPI_FLOAT,train_set,m * NUMROWS,MPI_FLOAT,0,MPI_COMM_WORLD);
		
		//将数据集划分成输入和输出
#if (ELM_TYPE == REGRESSION_TRAINING)
		for(i = 0;i<m*NUMROWS;i++){
			if(i % NUMROWS == 0){
				T[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
#else
		InitMatrix(T,m,OUTPUT_NEURONS,-1);
		for(i = 0;i < m*NUMROWS ;i++)
		{
			if(i % NUMROWS == 0){
				tempT[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
		for(i = 0;i < m ;i++)
		{
			k = tempT[i];
			if(k < OUTPUT_NEURONS && k >= 0){
				T[k+i*OUTPUT_NEURONS] = 1;
			}
		}
#endif
		//input * weight + B;
		MultiplyMatrix_cblas_s(input,m,INPUT_NEURONS,weight,INPUT_NEURONS,HIDDEN_NEURONS,tempI);
		AddMatrix_bais_s(tempI,bias,m,HIDDEN_NEURONS);
		//sigmoid
		SigmoidHandle_s(tempI,m,HIDDEN_NEURONS);
		TranspositionMatrix_s(tempI,tranpH,m,HIDDEN_NEURONS);
		//H'H
		MultiplyMatrix_cblas_s(tranpH,HIDDEN_NEURONS,m,T,m,OUTPUT_NEURONS,Ht);
		//HT
		MultiplyMatrix_cblas_s(tranpH,HIDDEN_NEURONS,m,tempI,m,HIDDEN_NEURONS,Hh);

		//接收子进程的数据，1、接收H'H,通过累加，2、H'T通过累加
		for(i = 1;i < size;i++){
			MPI_Recv(temphh,HIDDEN_NEURONS * HIDDEN_NEURONS,MPI_FLOAT,i,0,MPI_COMM_WORLD,&status);
			MPI_Recv(tempht,HIDDEN_NEURONS * OUTPUT_NEURONS,MPI_FLOAT,i,1,MPI_COMM_WORLD,&status);
			AddMatrix(tempht,Ht,HIDDEN_NEURONS * OUTPUT_NEURONS);
			AddMatrix(temphh,Hh,HIDDEN_NEURONS * HIDDEN_NEURONS);
		}
		//1、H'H累加的结果求解其逆
		//岭回归
		for(i = 0;i < HIDDEN_NEURONS*HIDDEN_NEURONS;i++){
				if(i % HIDDEN_NEURONS == i / HIDDEN_NEURONS) 
					Hh[i] += LUMMA; 
		}
		InverseMatirx_cblas_s(Hh,HIDDEN_NEURONS);
		//2、将上面两个结果相乘得到最终结果
		MultiplyMatrix_cblas_s(Hh,HIDDEN_NEURONS,HIDDEN_NEURONS,Ht,HIDDEN_NEURONS,OUTPUT_NEURONS,result);
		endtime = MPI_Wtime();
		printf("finish trainning...\n");
		printf("trainning time:%f\n",endtime - starttime);
		//SaveMatrix_s(result,"./result/result",HIDDEN_NEURONS,OUTPUT_NEURONS);
	
		/*---------------------------------------TEST------------------------------------------------------------*/	
#if (ELM_TYPE == REGRESSION_TRAINING)	
		printf("begin test...\n");
		printf("direction = %s\n",TEST_DATASET);
		if(LoadMatrix_s(train_set,TEST_DATASET,m,NUMROWS,1) == 0){
			printf("rank %d:load input file error!!!\n",rank);
			MPI_Abort(MPI_COMM_WORLD,-1);
		}
		k = 0;
		j = 0;
		for(i = 0;i<m*NUMROWS;i++){
			if(i % NUMROWS == 0){
				T[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
		MultiplyMatrix_cblas_s(input,m,INPUT_NEURONS,weight,INPUT_NEURONS,HIDDEN_NEURONS,tempI);
		AddMatrix_bais_s(tempI,bias,m,HIDDEN_NEURONS);
		//sigmoid
		SigmoidHandle_s(tempI,m,HIDDEN_NEURONS);
		//TranspositionMatrix_s(tempI,tranpH,m,HIDDEN_NEURONS);
		MultiplyMatrix_cblas_s(tempI,m,HIDDEN_NEURONS,result,HIDDEN_NEURONS,OUTPUT_NEURONS,Y);
		double MSE = 0,TrainingAccuracy;	
		for(i = 0;i< m;i++)
		{
			MSE += (Y[i] - T[i])*(Y[i] - T[i]);
		}
		TrainingAccuracy = sqrt(MSE/m);
		printf("Regression/trainning accuracy :%f\n",TrainingAccuracy);
#else
		if(LoadMatrix_s(train_set,TEST_DATASET,m,NUMROWS,1) == 0){
			printf("rank %d:load input file error!!!\n",rank);
			MPI_Abort(MPI_COMM_WORLD,-1);
		}
		InitMatrix(T,m,OUTPUT_NEURONS,-1);
		k = 0;
		j = 0;
		for(i = 0;i < m*NUMROWS ;i++)
		{
			if(i % NUMROWS == 0){
				tempT[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
		for(i = 0;i < m ;i++)
		{
			k = tempT[i];
			if(k < OUTPUT_NEURONS && k >= 0){
				T[k+i*OUTPUT_NEURONS] = 1;
			}
		}
		MultiplyMatrix_cblas_s(input,m,INPUT_NEURONS,weight,INPUT_NEURONS,HIDDEN_NEURONS,tempI);
		AddMatrix_bais_s(tempI,bias,m,HIDDEN_NEURONS);
		//sigmoid
		SigmoidHandle_s(tempI,m,HIDDEN_NEURONS);
		MultiplyMatrix_cblas_s(tempI,m,HIDDEN_NEURONS,result,HIDDEN_NEURONS,OUTPUT_NEURONS,Y);
		float MissClassificationRate_Training=0,TrainingAccuracy;
		double maxtag1,maxtag2;
		int tag1 = 0,tag2 = 0;
		for (i = 0; i < m; i++) {
			maxtag1 = Y[i * OUTPUT_NEURONS];
			tag1 = 0;
			maxtag2 = T[i * OUTPUT_NEURONS];
			tag2 = 0;
			for (j = i * OUTPUT_NEURONS + 1; j < i * OUTPUT_NEURONS + OUTPUT_NEURONS ; j++) {
				if(Y[j] > maxtag1){
					maxtag1 = Y[j];
					tag1 = j;
				}
				if(T[j] > maxtag2){
					maxtag2 = T[j];
					tag2 = j;
				}
			}
			if(tag1 != tag2)
		    	MissClassificationRate_Training ++;
		}
		TrainingAccuracy = 1 - MissClassificationRate_Training*1.0f/m;
		printf("Classification/trainning accuracy :%f\n",TrainingAccuracy);
#endif
	}else{
		printf("rank = %d\n",rank);
		int i,j = 0,k = 0;
		float *train_set,*T,*input,*weight,*bias,*tempI,*Ht,*Hh,*tranpH;
		train_set = (float *)calloc(m * NUMROWS,sizeof(float)); 				/* m * NUMROWS */
		T = (float *)calloc(m * OUTPUT_NEURONS,sizeof(float)); 					/* m * OUTPUT_NEURONS */
		input = (float *)calloc(m * INPUT_NEURONS,sizeof(float)); 				/* m * INPUT_NEURONS */
		weight = (float *)calloc(INPUT_NEURONS * HIDDEN_NEURONS,sizeof(float)); /* INPUT_NEURONS * HIDDEN_NEURONS */
		bias = (float *)calloc(HIDDEN_NEURONS,sizeof(float)); 					/* HIDDEN_NEURONS */
		tempI = (float *)calloc(m * HIDDEN_NEURONS,sizeof(float)); 				/* m * HIDDEN_NEURONS */
		Ht = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float)); 	/* HIDDEN_NEURONS * OUTPUT_NEURONS */
		Hh = (float *)calloc(HIDDEN_NEURONS * HIDDEN_NEURONS,sizeof(float)); 	/* HIDDEN_NEURONS * HIDDEN_NEURONS */
		tranpH = (float *)calloc(HIDDEN_NEURONS * m,sizeof(float)); 			/* m * HIDDEN_NEURONS */
#if (ELM_TYPE == CLASSIFICATION_TRAINING)
		float *tempT = (float *)calloc(m,sizeof(float));
#endif
		MPI_Bcast(weight,(INPUT_NEURONS)*(HIDDEN_NEURONS),MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(bias,HIDDEN_NEURONS,MPI_FLOAT,0,MPI_COMM_WORLD);
		
		MPI_Scatter(NULL,m * NUMROWS,MPI_FLOAT,train_set,m * NUMROWS,MPI_FLOAT,0,MPI_COMM_WORLD);
		if(rank == 1){
			printf("rank %d\n",rank);
			SaveMatrix_s(train_set,"./result/1",m,NUMROWS);
		}
		if(rank == 3){
			printf("rank %d\n",rank);
			SaveMatrix_s(train_set,"./result/3",m,NUMROWS);
		}
		/*将数据集划分成输入和输出*/
#if (ELM_TYPE == REGRESSION_TRAINING)
		for(i = 0;i<m*NUMROWS;i++){
			if(i % NUMROWS == 0){
				T[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
#else
		InitMatrix(T,m,OUTPUT_NEURONS,-1);
		for(i = 0;i < m*NUMROWS ;i++)
		{
			if(i % NUMROWS == 0){
				tempT[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
		for(i = 0;i < m ;i++)
		{
			k = tempT[i];
			if(k < OUTPUT_NEURONS && k >= 0){
				T[k+i*OUTPUT_NEURONS] = 1;
			}
		}
#endif
		//input * weight + B;
		MultiplyMatrix_cblas_s(input,m,INPUT_NEURONS,weight,INPUT_NEURONS,HIDDEN_NEURONS,tempI);
		AddMatrix_bais_s(tempI,bias,m,HIDDEN_NEURONS);
		//sigmoid
		SigmoidHandle_s(tempI,m,HIDDEN_NEURONS);
		TranspositionMatrix_s(tempI,tranpH,m,HIDDEN_NEURONS);
		//H'H
		MultiplyMatrix_cblas_s(tranpH,HIDDEN_NEURONS,m,T,m,OUTPUT_NEURONS,Ht);
		//HT
		MultiplyMatrix_cblas_s(tranpH,HIDDEN_NEURONS,m,tempI,m,HIDDEN_NEURONS,Hh);

		//发送ht,和Hh给主进程；
		MPI_Send(Hh,HIDDEN_NEURONS * HIDDEN_NEURONS,MPI_FLOAT,0,0,MPI_COMM_WORLD);
		MPI_Send(Ht,HIDDEN_NEURONS * OUTPUT_NEURONS,MPI_FLOAT,0,1,MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
