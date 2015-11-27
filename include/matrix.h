#ifndef MATRIX_H__
#define MATRIX_H__
void InitMatrix(float *matrix,int row,int column,float data);

void FreeMatrix(float **matrix,int row);
/*matrix add*/
int AddMatrix_bais(float **matrix,float *biase,int row,int column);
/*matrix transpose*/
int TranspositionMatrix(float** matrix, float** transpositionMatirx, int row, int column);
/*matrix inverse*/
int InverseMatirx(float **A, float **B, int n);
/*matrix multi*/
int MultiplyMatrix(float** matrix1, int row1, int column1, float** matrix2, int row2, int column2, float** matrix3);
/*pseudo inverse matrix*/
void PseudoInverseMatrix(float **matrix,int row,int column,float **PIMatrix);
/*use CLAS library*/
void InverseMatirx_clas(float **tempH,int column);
/*save the result to the file*/
int SaveMatrix(float **matrix,char *filepath,int row,int column);
/*load the sample from file in matrix*/
int LoadMatrix(float **matrix,char *filepath,int row,int column,int type);
int LoadMatrix_s(float *matrix,char *filepath,int row,int column,int type);
int TranspositionMatrix_s(float *matrix, float *transpositionMatirx, int row, int column);
void InverseMatirx_cblas_s(float *H,int row);
void MultiplyMatrix_cblas_s(float *matrix1,int row1,int column1,float *matrix2,int row2,int column2,float *matrix3);
int AddMatrix(float *matrix1,float *matrix2,int n);
int AddMatrix_bais_s(float *matrix,float *biase,int row,int column);
int SaveMatrix_s(float *matrix,char *filepath,int row,int column);
#endif
