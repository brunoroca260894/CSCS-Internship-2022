#include <cuda_runtime.h>

#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include <cmath>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#define prec_save 7

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

    const int N = 5;

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    /***********************/
    /* SETTING THE PROBLEM */
    /***********************/
    // --- Setting the host, N x N matrix
    double *h_A = (double *)malloc(N * N * sizeof(double));
    setPDMatrix(h_A, N);
    
    // --- write original matrix into file 
    // saveCPUrealtxt(h_A, "/home/bruno/Documents/internship/example_cublas/h_A_original.txt", int(N * N));

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
    cudaMalloc(&work, work_size * sizeof(double));
    cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, devInfo);
    
    int devInfo_h = 0;  
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (devInfo_h != 0) 
    	std::cout << "Unsuccessful cholesky execution " << "devInfo = " << devInfo_h << "\n\n";
    else
	std::cout << "Successful cholesky execution " << "devInfo = " << devInfo_h << "\n\n";

    // --- At this point, the lower triangular part of A contains the elements of L.       
    // --- KEEP IN MIND cublas stores the matrix in column-major. BE CAREFUL how to access matrix elements 
    
    /***************************************/
    /* CHECKING THE CHOLESKY DECOMPOSITION */
    /***************************************/
    const char *path_h="/home/bruno/Documents/internship/example_cublas/host_A.txt";
    const char *path_d="/home/bruno/Documents/internship/example_cublas/device_A.txt";
    const char *path_T="/home/bruno/Documents/internship/example_cublas/triangular.txt";
    const char *path_product="/home/bruno/Documents/internship/example_cublas/product.txt"; //this files stores A = L*L^T.        
       
    std::cout << "AFTER WRITTING INTO FILE \n";
    // allocate host memory for triangular matrix and product of triangular matrices
    double *h_T = (double *)malloc(N * N * sizeof(double));   
    double *h_product = (double *)malloc(N * N * sizeof(double));
    memset(h_T, 0, N * N * sizeof(double));
    
    // extract triangular matrix and perform product of triangular matrices
    extractMultiplyTriangular(d_A, h_T, h_product, N);
    std::cout << "AFTER TRIANGULAR RESULTS\n";
    
    saveCPUrealtxt(h_A, path_h, N * N);
    saveGPUrealtxt(d_A, path_d, N * N);
    saveCPUrealtxt(h_T, path_T, N * N);
    saveCPUrealtxt(h_product, path_product, N * N);
    std::cout << "AFTER SAVING RESULTS\n";        
    
    cusolverDnDestroy(solver_handle);
    free(h_A);
    free(h_T);
    free(h_product);
    
    cudaFree(d_A);  
    
    std::cout << "--PROGRAM EXECUTED SUCCESSFULLY\n"; 
    return 0;

}
