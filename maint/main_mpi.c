#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mpi.h"
#include "elm.h"
#include "matrix.h"

int main(int argc,char **argv){
	int m;
	int rank,size; 
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//将样本集均等划分，主进程不处理样本，子进程处理样本
	m = DATASET/(size-1);
	printf(" m = %d\n",m);
	if(rank == 0){
		int i;
		float *Ht,*Hh,*tempht,*temphh,*result;
		result = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float));	/* HIDDEN_NEURONS * OUTPUT_NEURONS */
		Ht = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float)); 		/* HIDDEN_NEURONS * OUTPUT_NEURONS */
		tempht = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float)); 	/* HIDDEN_NEURONS * OUTPUT_NEURONS */
		Hh = (float *)calloc(HIDDEN_NEURONS * HIDDEN_NEURONS,sizeof(float)); 		/* HIDDEN_NEURONS * HIDDEN_NEURONS */
		temphh = (float *)calloc(HIDDEN_NEURONS * HIDDEN_NEURONS,sizeof(float)); 	/* HIDDEN_NEURONS * HIDDEN_NEURONS */
		//初始化
		/*权重在计算过程中需要进行转置，因为参数全是随机，所以为了方便计算，
		开始定义就进行转置，原行列为HIDDEN_NEURONS*INPUT_NEURONS*/
		float *weight = (float *)calloc(INPUT_NEURONS*HIDDEN_NEURONS,sizeof(float)); /*INPUT_NEURONS * HIDDEN_NEURONS */
		float *bias = (float *)calloc(HIDDEN_NEURONS,sizeof(float)); 
		RandomWeight_s(weight,INPUT_NEURONS,HIDDEN_NEURONS);
		RandomBiase(bias,HIDDEN_NEURONS);
		//随机权重和偏置，并将权重和偏置广播给所有子进程
		printf("file:%s,func:%s,line:%d rank:%d\n",__FILE__,__func__,__LINE__,rank);
		MPI_Bcast(weight,INPUT_NEURONS * HIDDEN_NEURONS,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(bias,HIDDEN_NEURONS,MPI_FLOAT,0,MPI_COMM_WORLD);
		//接收子进程的数据，1、接收H'H,通过累加，2、H'T通过累加
		for(i = 1;i < size;i++){
			MPI_Recv(Hh,HIDDEN_NEURONS * HIDDEN_NEURONS,MPI_FLOAT,i,0,MPI_COMM_WORLD,&status);
			MPI_Recv(Ht,HIDDEN_NEURONS * OUTPUT_NEURONS,MPI_FLOAT,i,1,MPI_COMM_WORLD,&status);
			AddMatrix(Ht,tempht,HIDDEN_NEURONS * OUTPUT_NEURONS);
			AddMatrix(Hh,temphh,HIDDEN_NEURONS * HIDDEN_NEURONS);
		}
		//1、H'H累加的结果求解其逆
		//岭回归
		for(i = 0;i < HIDDEN_NEURONS*HIDDEN_NEURONS;i++){
				if(i % HIDDEN_NEURONS == i / HIDDEN_NEURONS) 
					Hh[i] += 1; 
		}
		InverseMatirx_cblas_s(Hh,HIDDEN_NEURONS);
		//2、将上面两个结果相乘得到最终结果
		MultiplyMatrix_cblas_s(Hh,HIDDEN_NEURONS,HIDDEN_NEURONS,Ht,HIDDEN_NEURONS,OUTPUT_NEURONS,result);
		//回归准确率测试
		SaveMatrix_s(result,"./result/result",HIDDEN_NEURONS,OUTPUT_NEURONS);	
	}else{
		int i,j = 0,k = 0;
		char dir[20];
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
		MPI_Bcast(weight,(INPUT_NEURONS)*(HIDDEN_NEURONS),MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(bias,HIDDEN_NEURONS,MPI_FLOAT,0,MPI_COMM_WORLD);
		sprintf(dir,"./sample/p%d",rank);
		//从本地读取相应的样本集
		if(LoadMatrix_s(train_set,dir,m,NUMROWS) == 0){
			printf("rank %d:load input file error!!!\n",rank);
			MPI_Abort(MPI_COMM_WORLD,-1);
		}
		
		/*将数据集划分成输入和输出*/
		for(i = 0;i<m*NUMROWS;i++){
			if(i % NUMROWS == 0){
				T[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
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
