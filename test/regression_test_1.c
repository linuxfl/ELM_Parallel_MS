#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/matrix.h"
#include "../include/elm.h"
#define TESTDATA 1000

int main(int argc,char **argv)
{

		int i,j = 0,k = 0;
		float MSE = 0.0,TrainingAccuracy;
		float *train_set,*T,*input,*weight,*bias,*tempI,*tranpH,*out,*Y;
		train_set = (float *)calloc(TESTDATA * NUMROWS,sizeof(float)); 				/* m * NUMROWS */
		T = (float *)calloc(TESTDATA * OUTPUT_NEURONS,sizeof(float)); 					/* m * OUTPUT_NEURONS */
		input = (float *)calloc(TESTDATA * INPUT_NEURONS,sizeof(float)); 				/* m * INPUT_NEURONS */
		weight = (float *)calloc(INPUT_NEURONS * HIDDEN_NEURONS,sizeof(float)); /* INPUT_NEURONS * HIDDEN_NEURONS */
		bias = (float *)calloc(HIDDEN_NEURONS,sizeof(float)); 					/* HIDDEN_NEURONS */
		tempI = (float *)calloc(TESTDATA * HIDDEN_NEURONS,sizeof(float)); 				/* m * HIDDEN_NEURONS */
		tranpH = (float *)calloc(HIDDEN_NEURONS * TESTDATA,sizeof(float)); 			/* m * HIDDEN_NEURONS */
		out = (float *)calloc(HIDDEN_NEURONS * OUTPUT_NEURONS,sizeof(float)); 			/* m * HIDDEN_NEURONS */
		Y = (float *)calloc(TESTDATA * OUTPUT_NEURONS,sizeof(float)); 			/* m * HIDDEN_NEURONS */
		printf("begin to test...\n");
		if(LoadMatrix_s(weight,"../result/weight",INPUT_NEURONS,HIDDEN_NEURONS) == 0){
			printf("load weight file error!!!\n");
			exit(-1);
		}
	
		if(LoadMatrix_s(bias,"../result/bias",1,HIDDEN_NEURONS) == 0){
			printf("load bias file error!!!\n");
			exit(-1);
		}

		//ŽÓ±ŸµØ¶ÁÈ¡ÏàÓŠµÄÑù±ŸŒ¯
		if(LoadMatrix_s(train_set,"../sample/1",TESTDATA,NUMROWS) == 0){
			printf("load input file error!!!\n");
			exit(-1);
		}
		/*œ«ÊýŸÝŒ¯»®·Ö³ÉÊäÈëºÍÊä³ö*/
		for(i = 0;i<TESTDATA*NUMROWS;i++){
			if(i % NUMROWS == 0){
				T[k++] = train_set[i];
			}else{
				input[j++] = train_set[i];
			}
		}
		//input * weight + B;
		MultiplyMatrix_cblas_s(input,TESTDATA,INPUT_NEURONS,weight,INPUT_NEURONS,HIDDEN_NEURONS,tempI);
		AddMatrix_bais_s(tempI,bias,TESTDATA,HIDDEN_NEURONS);
		//sigmoid
		SigmoidHandle_s(tempI,TESTDATA,HIDDEN_NEURONS);
		TranspositionMatrix_s(tempI,tranpH,TESTDATA,HIDDEN_NEURONS);

		if(LoadMatrix_s(out,"../result/result",HIDDEN_NEURONS,OUTPUT_NEURONS) == 0){
			printf("load result file error!!!\n");
			exit(-1);
		}
		MultiplyMatrix_cblas_s(tranpH,TESTDATA,HIDDEN_NEURONS,out,HIDDEN_NEURONS,OUTPUT_NEURONS,Y);
		for(i = 0;i< TESTDATA;i++)
		{
			MSE += (Y[i] - T[i])*(Y[i] - T[i]);
		}
		TrainingAccuracy = sqrt(MSE/TESTDATA);
	
		printf("trainning accuracy :%f\n",TrainingAccuracy);
	
		printf("test complete...\n");
		return 0;
}	
