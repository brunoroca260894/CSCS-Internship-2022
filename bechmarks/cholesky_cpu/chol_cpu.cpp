#include<iostream>
#include<fstream>
#include<string.h>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include <cmath>
#include <vector>
#include<chrono>

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
/* SAVE MEASURED TIME INTO ARRAY */
/************************************/
template <class T>
void saveData(const T * A, const char *filename, const int cols, const int runs) {
		
    std::ofstream outfile;
    outfile.open(filename);
    for(int i=0; i < runs; i++)
    {
    	for(int j=0; j<cols; j++)
    	{
    		outfile << std::setprecision(prec_save) << std::left << std::setw(12) << A[i*cols+j] ;
    	}
    	outfile << "\n";
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
	//for cholesky decomposition
  	int info;
  	// store upper part
	char uplo = 'U';
	int major = 1;
	
	int matrix_sizes [] = { 100, 500, 1000, 5000, 10000};
    int total_N = int ( sizeof(matrix_sizes)/sizeof(matrix_sizes[0]) );
    int total_runs = 10;
    int N;
    double *times= (double *)malloc(total_N * total_runs * sizeof(double));

	for(int i= 0; i < total_runs; i ++)
    {    	
	    std::cout << "run: " << i +1 <<std::endl;
		for(int j = 0; j< total_N; j++)
		{
			std::cout << "matrices size: " << matrix_sizes[j] <<std::endl;
			N = matrix_sizes[j];
			// dpotrf_ and dpotri_ expect a real symmetric positive definite matrix 
			// in a vector ordered by columns
			double *A = (double *)malloc(N * N * sizeof(double));
		  	
		  	// fill in matrices with random data
		  	setPDMatrix(A, N);
		  	
		  	// compute Cholesky factorization
		  	auto begin = std::chrono::high_resolution_clock::now();
		  	dpotrf_(&uplo, &N, A, &N, &info);
		  	auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
			printf("Time measured: %.10f seconds.\n", elapsed.count() * 1e-6);
			times[i*total_N +j] = elapsed.count();
			
		  	if (info != 0) std::cout << "Error in dpotrf_(): Flag is " << info << std::endl;
		  	
		  	free(A);
	  	}
	  	std::cout << "------------------" << "\n\n";
	}
	
	const char *filename="/home/bruno/Documents/internship/benchmarks_cpu_gpu/cholesky_cpu/time_matrices_cpu.txt";
    saveData(times, filename, total_N, total_runs);
    free(times);
    std::cout << "--PROGRAM EXECUTED SUCCESSFULLY\n";
  	return 0;
}

