/*
	matrix operation
	2015.9.15 fangling
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"
#include "lapacke.h"
#include <omp.h>
#include "../include/matrix.h"

void InitMatrix(float *matrix,int row,int column,float data)
{
	int i;
	for(i = 0;i < row*column;i++)
	{
		matrix[i] = data;
	}
}

int LoadMatrix_s(float *matrix,char *filepath,int row,int column,int type)
{
	int i,j;
	size_t len = 0;
	float temp;
	FILE *fp;
	char *line = NULL;
	fp = fopen(filepath,"r+b");
	if(!fp)
		return 0;
	if(type == 1){
		//for(i=0;i<column;i++)
		//{
			//fscanf(fp,"%f",&temp);
			getline(&line, &len, fp);
		//}
		if(line != NULL)
			free(line);
	}
	for(i = 0;i < row;i++)
	{
		for(j = 0;j < column;j++)
		{
			fscanf(fp,"%f",&temp);
			matrix[i*column+j] = (float)temp;
		}
	}
	
	fclose(fp);
	return 1;
}

int LoadMatrix(float **matrix,char *filepath,int row,int column,int type)
{
	int i,j;
	size_t len = 0;
	float temp;
	FILE *fp;
	char *line = NULL;
	fp = fopen(filepath,"r+b");
	if(!fp)
		return 0;
	if(type == 1){
		//for(i=0;i<column;i++)
		//{
			//fscanf(fp,"%f",&temp);
			getline(&line, &len, fp);
		//}
		if(line != NULL)
			free(line);
	}
	for(i = 0;i < row;i++)
	{
		for(j = 0;j < column;j++)
		{
			fscanf(fp,"%f",&temp);
			matrix[i][j] = (float)temp;
		}
	}
	fclose(fp);
	return 1;
}

int SaveMatrix(float **matrix,char *filepath,int row,int column)
{
	int i,j;
	FILE *fp;
	fp = fopen(filepath,"w+b");
	if(!fp)
		return 0;
	for(i=0;i<row;i++){
		for(j=0;j<column;j++)
		{
			fprintf(fp,"%f ",matrix[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	return 1;
}
int SaveMatrix_s(float *matrix,char *filepath,int row,int column)
{
	int i,j;
	FILE *fp;
	fp = fopen(filepath,"w+b");
	if(!fp)
		return 0;
	for(i=0;i<row;i++){
		for(j=0;j<column;j++)
		{
			fprintf(fp,"%f ",matrix[i*column+j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	return 1;
}
void FreeMatrix(float **matrix,int row)
{
	int i;
	for(i=0;i<row;i++)
	{
		free((void *)matrix[i]);
	}
	free((void *)matrix);
	matrix = NULL;
}

/*matrix transpose*/

int TranspositionMatrix(float** matrix, float** transpositionMatirx, int row, int column)
{
	int i;
	int j;

	if(matrix == NULL || transpositionMatirx == NULL || row <= 0 || column <= 0)
		return 1;

	for(i = 0; i < row; i++)
	{
		for(j = 0; j < column; j++)
		{
			transpositionMatirx[j][i] = matrix[i][j];
		}
	}
	return 0;
}

int TranspositionMatrix_s(float *matrix, float *transpositionMatirx, int row, int column)
{
	int i,j;
	//int r,c;
	if(matrix == NULL || transpositionMatirx == NULL || row <= 0 || column <= 0)
		return 1;
	float **matrix1,**matrix2;
	matrix1 = (float **)calloc(row,sizeof(float *));
	matrix2 = (float **)calloc(column,sizeof(float *));

	for(i = 0;i< row;i++)
		matrix1[i] = (float *)calloc(column,sizeof(float));
	for(i = 0;i< column;i++)
		matrix2[i] = (float *)calloc(row,sizeof(float));
	/*for(i = 0;i < row*column;i++)
	{
		r = i/column;
		c = i%column;
		transpositionMatirx[i] = matrix[r*row+c];
	}*/
	for(i = 0;i<row;i++){
		for(j = 0;j < column;j++)
		{
			matrix1[i][j] = matrix[i*column+j];
		}
	}
	TranspositionMatrix(matrix1,matrix2,row,column);
	for(i = 0;i<column;i++){
		for(j = 0;j < row;j++)
		{
			transpositionMatirx[i*row+j] = matrix2[i][j];
		}
	}
	FreeMatrix(matrix1,row);
	FreeMatrix(matrix2,column);
	return 0;
}

void InverseMatirx_clas(float **H,int row)
{
	int i,j;
	int lda = row;
	int *ipiv;
	ipiv = (int *)calloc(row,sizeof(int));
	float *temp;
	temp = (float *)calloc(row*row,sizeof(float));
	for(i = 0;i < row;i++)
	{
		for(j = 0 ;j<row;j++)
		{
			temp[i*row+j] = H[i][j];
		}
	}
	LAPACKE_sgetrf(LAPACK_ROW_MAJOR,row,row,temp,lda,ipiv);
	LAPACKE_sgetri(LAPACK_ROW_MAJOR,row,temp,lda,ipiv);
	for(i = 0;i < row;i++)
	{
		for(j = 0 ;j<row;j++)
		{
			H[i][j] = temp[i*row+j]; 
		}
	}
}

void InverseMatirx_cblas_s(float *H,int row)
{
	int lda = row;
	int *ipiv;
	ipiv = (int *)calloc(row,sizeof(int));
	LAPACKE_sgetrf(LAPACK_ROW_MAJOR,row,row,H,lda,ipiv);
	LAPACKE_sgetri(LAPACK_ROW_MAJOR,row,H,lda,ipiv);
}

void MultiplyMatrix_cblas(float **matrix1,int row1,int column1,float **matrix2,int row2,int column2,float **matrix3)
{
	int i,j,k = 0;
	float *A,*B,*C;
	A = (float *)calloc(row1*column1,sizeof(float));
	B = (float *)calloc(row2*column2,sizeof(float));
	C = (float *)calloc(row1*column2,sizeof(float));

	for(i = 0;i < row1;i++){
		for(j = 0;j < column1;j++)
		{
			A[k++] = matrix1[i][j];
		}
	}
	k = 0;
	for(i = 0;i < row2;i++)
	{
		for(j = 0;j < column2;j++)
			B[k++] = matrix2[i][j];
	}
		
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,row1,column2,column1,1,A,column1,B,column2,0,C,column2);

	k = 0;
	
	for(i = 0;i < row1;i++)
	{
		for(j = 0;j < column2;j++)
			matrix3[i][j] = C[k++];
	}
	free(A);
	free(B);
	free(C);
}

void MultiplyMatrix_cblas_s(float *matrix1,int row1,int column1,float *matrix2,int row2,int column2,float *matrix3)
{
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,row1,column2,column1,1,matrix1,column1,matrix2,column2,0,matrix3,column2);
}

int AddMatrix(float *matrix1,float *matrix2,int n)
{
	int i;
	for(i=0;i<n;i++){
		matrix2[i] += matrix1[i];
	}
	return 0;
}

int AddMatrix_bais(float **matrix,float *biase,int row,int column)
{
	int i,j;
	for(i=0;i<row;i++){
		for(j=0;j<column;j++){
			matrix[i][j] += biase[i];
		}
	}
	return 0;
}

int AddMatrix_bais_s(float *matrix,float *bias,int row,int column)
{
	int i,j = 0;
	for(i=0;i<row*column;i++){
		if(j == column)
			j = 0;
		matrix[i] += bias[j++];
	}
	return 0;
}

int MultiplyMatrix(float** matrix1, int row1, int column1, float** matrix2, int row2, int column2, float** matrix3)
{
	int i,j,k;
	if(column1 != row2)
		return 1;
	
	for(i = 0; i < row1; i++)
	{
		for(j = 0; j < column2; j++)
		{
			for(k = 0; k < column1; k++)
			{
				matrix3[i][j] += matrix1[i][k]*matrix2[k][j];
			}
		}
	}
	
	return 0;
}

void PseudoInverseMatrix(float **matrix,int row,int column,float **PIMatrix)
{
	int i,j;
	float **tempH,**tranposeH;
	tranposeH = (float **)calloc(column,sizeof(float *));
	tempH = (float **)calloc(column,sizeof(float *));
	
	for(i = 0;i<column;i++)
		tranposeH[i] = (float *)calloc(row,sizeof(float));
	
	for(i = 0;i<column;i++){
		tempH[i] = (float *)calloc(column,sizeof(float));
	}

	printf("begin to compute pseudoinverse matrix step 1...\n");
	TranspositionMatrix(matrix,tranposeH,row,column);
	
	printf("begin to compute pseudoinverse matrix step 2...\n");
	MultiplyMatrix_cblas(tranposeH,column,row,matrix,row,column,tempH);
	printf("begin to compute pseudoinverse matrix step 3...\n");
	for(i = 0;i < column;i++){
		for(j = 0;j < column;j++)
		{
			if(i == j)
				tempH[i][j] += 1; 
		}
	}
	SaveMatrix(tempH,"./result/tempH",column,column);
	InverseMatirx_clas(tempH,column);
	SaveMatrix(tempH,"./result/invtempH",column,column);
	//printf("the time of inverse matrix is  %f\n",endtime-starttime);
	printf("begin to compute pseudoinverse matrix step 4...\n");
	MultiplyMatrix_cblas(tempH,column,column,tranposeH,column,row,PIMatrix);
}
