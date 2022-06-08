#include<iostream>
#include<fstream>
#include<string.h>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include <cmath>
#include <vector>

#define prec_save 8

extern "C" int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" int dpotri_(char *uplo, int *n, double *a, int *lda, int *info);

extern "C" void dgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, double *ALPHA, double *A, const int *LDA, double *B, const int *LDB, double *BETA, double *C, const int *LDC);

/******************************************/
/* define real SPD matrix. This matrix is */
/* filled with random numbers */
/******************************************/
// --- Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void setPDMatrix(double * __restrict h_A, const int N) {

    // --- Initialize random seed
    srand(time(NULL));

    double *h_A_temp = (double *)malloc(N * N * sizeof(double));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_A_temp[i * N + j] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) 
            h_A[i * N + j] = 0.5 * (h_A_temp[i * N + j] + h_A_temp[j * N + i]);

    for (int i = 0; i < N; i++) h_A[i * N + i] = h_A[i * N + i] + N;

}

template <typename T>
void extractTriangular(const T* A, T* L, const int N);

template <typename T>
void extractTriangular(const T* A, T* L, const int N)
{
	for(int i=0; i < N; i++) 
    {
    	for(int j=0; j < N; j++)
    	{
    		if(i>=j) *(L + N*i + j) = *(A + N*i + j);               
        }
    } 
}

/************************************/
/* SAVE REAL ARRAY FROM CPU TO FILE */
/************************************/
template <class T>
void saveCPUrealtxt(const T * h_in, const char *filename, const int M) {
		
    int N = int(sqrt(M));
    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < N; i++) 
    {
    	for(int j=0; j<N; j++)
    	{
    	    if((j%N==0) && (i>0) )    	
	    {
	    	outfile << "\n";
	    }
	    outfile << std::setprecision(prec_save) << h_in[i+j*N] << "\t\t";
    	}    	    	
    }           	
    outfile.close();
}

/************************************/
/* PRINT MATRIX, ROW OR COLUMN MAJOR*/
/* 1 = ROW MAJOR, 0 = COLUMN MAJOR  */
/************************************/
template <class T>
void print_squareMatrix(T *A, int N, int *major)
{
	int index = 0;
	for(int i=0; i <N; i++)
	{
		for(int j=0; j <N; j++)
		{
			if(*major == 1)
			{
				index = i*N+j;
			}
			else
			{
				index = j*N+i;
			}
			std::cout << std::setprecision(prec_save) << std::right <<std::setw(10) << A[index] <<"\t"  ;		
	    }
	    std::cout << "\n";
	}
}

/************************************/
/* MAIN FUNCTION */
/************************************/
int main()
{	
//size of square matrix
	int N = 8;  
	// for matrix product 
  	//tranpose both matrices due to column major arragement
  	char TRANS = 'T'; 
  	char TRANS_L = 'N';
  	int M = N;
  	int K = N;
  	double ALPHA = 1.0;
  	int LDA = N;
  	int LDB = N;
  	double BETA = 0.0;
  	int LDC = N; 
  	int major_A = 1;
  	int major_C = 0;
	
	//for cholesky decomposition
  	int info;
  	// store upper part
	char uplo = 'u';
 
	// dpotrf_ and dpotri_ expect a real symmetric positive definite matrix 
	// in a vector ordered by columns
	double *A = (double *)malloc(N * N * sizeof(double));
  	double *B = (double *)malloc(N * N * sizeof(double));
  	double *C = (double *)malloc(N * N * sizeof(double));
  	double *L = (double *)malloc(N * N * sizeof(double));
  	
  	memset(L, 0, N * N * sizeof(double));
  	
  	// fill in matrices with random data
  	setPDMatrix(B, N);
  	setPDMatrix(A, N);
  	
  	//print original matrix
  	std::cout << "original matrix A: "<< std::endl; 
  	int major = 1;
  	print_squareMatrix(A, N, &major);

  	// compute Cholesky factorization
  	// FUNCTION: dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
  	
  	dpotrf_(&uplo, &N, A, &N, &info);
  	if (info != 0) std::cout << "Error in dpotrf_(): Flag is " << info << std::endl;
  	
  	//extract L factor
  	extractTriangular(A, L, N);
  	
  	//print modified matrix A
  	std::cout << "modified matrix A (lower part): "<< std::endl; 
  	major = 1;
  	print_squareMatrix(A, N, &major);   
  	
  	//print cholesky factor L
  	std::cout << "cholesky factor L: "<< std::endl; 
  	major = 1;
  	print_squareMatrix(L, N, &major);   	
  	
  	// here, we pass L^T * L due to column major 
  	dgemm_(&TRANS, &TRANS_L, &M, &N, &K, &ALPHA, L, &LDA, L, &LDB, &BETA, C, &LDC);
  	//print product A = LL^T matrix
  	std::cout << " matrix A = LL^T: "<< std::endl; 
  	major = 1;
  	print_squareMatrix(C, N, &major);
  	
  	// use Cholesky factorization to compute the inverse
  	dpotri_(&uplo,&N, A, &N, &info);
  	if (info != 0) std::cout << "Error in dpotri_(): Flag is " << info << std::endl;
  
  	return 0;
}

