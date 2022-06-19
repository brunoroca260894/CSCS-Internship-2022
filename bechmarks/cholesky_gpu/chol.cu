#include<cuda_runtime.h>
#include<cusolverDn.h>
#include<cublas_v2.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<cmath>
#include<chrono>

#define prec_save 10

/******************************************/
/* this code is based on:
/*  https://stackoverflow.com/questions/29196139/cholesky-decomposition-with-cuda */
/* https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/ */
/******************************************/

/******************************************/
/* define real SPD matrix. This matrix is */
/* filled with random numebers */
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

/******************************************/
/* MATRIX MULTIPLICATION USING cuBLAS */
/******************************************/
// C(m,n) = A(m,k) * B(k,n) -> for general size matrices
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

/******************************************/
/* EXTRACT LOWER/UPPER TRIANGULAR MATRIX */
/* AND MULTIPLY TRIANGULAR MATRICES */
/******************************************/
template <typename T>
void extractMultiplyTriangular(const T* d_A, T* h_T, T* h_product, const int N)
{
     int totalElements = int(N * N);         
     T* h_A = (T*)malloc(totalElements* sizeof(T));             
     cudaMemcpy(h_A, d_A, totalElements * sizeof(T), cudaMemcpyDeviceToHost);
     
     T *d_T;
     cudaMalloc(&d_T, totalElements * sizeof(T));
     T *d_product;
     cudaMalloc(&d_product, totalElements * sizeof(T));
     
     for(int i=0; i < N; i++) 
     {
          for(int j=0; j < N; j++)
          {
               if(i<=j) 
               {                 
                    *(h_T + N*i + j) = *(h_A + N*i + j);                    
               }               
          }
     } 
     
     //copy diagonal matrix back to GPU
     cudaMemcpy(d_T, h_T, totalElements * sizeof(T), cudaMemcpyHostToDevice);
     
     // perform matrix multiplication A = U^{T} U     
     gpu_blas_mmul(d_T, d_T, d_product, N, N, N);
     cudaMemcpy(h_product, d_product, totalElements * sizeof(T), cudaMemcpyDeviceToHost);
     
     free(h_A);
     cudaFree(d_T);
     cudaFree(d_product);     
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
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {
    
    int N = int(sqrt(M));
    T *h_in = (T *)malloc(M * sizeof(T));

    cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost);

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

/********/
/* MAIN */
/********/
int main(){

    int matrix_sizes [] = { 100, 500, 1000, 5000, 10000};
    //int matrix_sizes [] = { 100, 500, 1000, 5000};
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
			// CUDA solver initialization
			cusolverDnHandle_t solver_handle;
			cusolverDnCreate(&solver_handle);

			// --- Setting the host, N x N matrix
			double *h_A = (double *)malloc(N * N * sizeof(double));
			setPDMatrix(h_A, N);
			
			// --- Allocate device space for the input matrix 
			double *d_A; 
			cudaMalloc(&d_A, N * N * sizeof(double));

			// --- Move the relevant matrix from host to device
			cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);

			/****************************************/
			/* COMPUTING THE CHOLESKY DECOMPOSITION */
			/****************************************/
			// --- cuSOLVE input/output parameters/arrays
			int work_size = 0;
			int *devInfo;           
			cudaMalloc(&devInfo, sizeof(int));

			// --- CUDA CHOLESKY initialization
			// to compute size of workspace
			cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &work_size);
			std::cout << "size of workspace "<< work_size << "\n";

			// --- CUDA POTRF execution. Actual Cholesky computation
			double *work;   
			cudaMalloc(&work, work_size * sizeof(double)); //store information on workspace
			
			// Here we compute the Cholesky factorization for varying matrix sizes			
			auto begin = std::chrono::high_resolution_clock::now();
			cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, devInfo);
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
			printf("Time measured: %.10f seconds.\n", elapsed.count() * 1e-6);
			times[i*total_N +j] = elapsed.count();
			
			int devInfo_h = 0;  
			cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
			
			if (devInfo_h != 0) 
				std::cout << "Unsuccessful cholesky execution " << "devInfo = " << devInfo_h << "\n";
			else
			std::cout << "Successful cholesky execution " << "devInfo = " << devInfo_h << "\n";
			
			cusolverDnDestroy(solver_handle);
			free(h_A);
			cudaFree(d_A); 			
		}
		std::cout << "------------------" << "\n\n";
    }
    
    const char *filename="/home/bruno/Documents/internship/benchmarks_cpu_gpu/cholesky_gpu/time_matrices.txt";
    saveData(times, filename, total_N, total_runs);
    
    free(times);
    std::cout << "--PROGRAM EXECUTED SUCCESSFULLY\n"; 
    return 0;

}
