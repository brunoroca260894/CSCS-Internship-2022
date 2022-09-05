#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "include/cublas_api_cholesky.h"
#include "include/miscellaneous_functions.h"

int main() {
  // parameters setup
  // number of rows and cols of matrix A
  int M = 10000;
  // number of rows in tiles of A
  int M_tile = 256;
  // create matrix on CPU
  auto begin = std::chrono::high_resolution_clock::now();
  double* h_A = (double*)malloc(M * M * sizeof(double));
  set_SPD_Matrix(h_A, M);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "total time to create matrix on CPU in seconds: "
            << elapsed.count() * 1e-6 << std::endl;
  // print_matrix(h_A, M, M);

  // create matrix on GPU
  begin = std::chrono::high_resolution_clock::
      now();  // time to create matrix on GPU, this includes data copy
  Matrix<double> d_A(h_A, M, M, M_tile, M_tile);  // create matrix on GPU
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "total time to create matrix on GPU in seconds: "
            << elapsed.count() * 1e-6 << std::endl;

  // --- CUBLAS initialization
  cublasHandle_t cublas_handle;  // this is for cublas functions
  cublasCreate(&cublas_handle);

  // --- CUDA solver initialization
  cusolverDnHandle_t solver_handle;  // this is for Cholesky decomposition
  cusolverDnCreate(&solver_handle);

  /****************************************/
  /* COMPUTING THE CHOLESKY DECOMPOSITION */
  /****************************************/
  int h_devInfo;
  double alpha_trsm = 1.0;
  double alpha = -1.0;
  double beta = 1;
  cusolverStatus_t cholStatus;
  cublasStatus_t dtrsmStatus;
  cublasStatus_t dsyrkStatus;
  cublasStatus_t dgemmStatus;

  int numTiles = d_A.tilesInRow;

  std::cout << "number of tiles along column direction " << numTiles
            << std::endl;

  begin = std::chrono::high_resolution_clock::
      now();  // time to create matrix on GPU, this includes data copy
  for (int k = 0; k < numTiles; k++) {
    cholStatus = cublas_dpotrf_tile(solver_handle, d_A(k, k), &h_devInfo);
    if (CUSOLVER_STATUS_SUCCESS != cholStatus || 0 != h_devInfo) {
      std::cout << "Unsuccessful cholesky execution " << std::endl;
      return 1;
    }

    for (int m = k + 1; m < numTiles; m++) {
      dtrsmStatus =
          cublas_dtrsm_tile(cublas_handle, d_A(k, k), d_A(m, k), &alpha_trsm);
      if (CUBLAS_STATUS_SUCCESS != dtrsmStatus) {
        std::cout << "dtrsm performed unsuccessfully " << std::endl;
        std::cout << "error " << dtrsmStatus << std::endl;
        return 1;
      }
    }

    for (int n = k + 1; n < numTiles; n++) {
      dsyrkStatus =
          cublas_dsyrk_tile(cublas_handle, d_A(n, k), d_A(n, n), &alpha, &beta);
      if (CUBLAS_STATUS_SUCCESS != dsyrkStatus) {
        std::cout << "dsyrk performed unsuccessfully " << std::endl;
        std::cout << "error " << dsyrkStatus << std::endl;
        return 1;
      }

      for (int m = n + 1; m < numTiles; m++) {
        dgemmStatus = cublas_dgemm_tile(cublas_handle, d_A(m, k), d_A(n, k),
                                        d_A(m, n), &alpha, &beta);
        if (CUBLAS_STATUS_SUCCESS != dsyrkStatus) {
          std::cout << "dgemm performed unsuccessfully " << std::endl;
          std::cout << "error " << dgemmStatus << std::endl;
          return 1;
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "total time to compute tiled Cholesky on the GPU in seconds: "
            << elapsed.count() * 1e-6 << std::endl;

  // copy modified matrix back to CPU
  cudaMemcpy(h_A, d_A(0, 0).firstElement, M * M * sizeof(double),
             cudaMemcpyDeviceToHost);
  extract_triangular(h_A, M);

  std::cout << "Cholesky factor L " << std::endl;
  std::cout << "L=" << std::endl;
  // print_matrix(h_A, M, M);

  // free resources
  free(h_A);
  cudaFree(d_A(0, 0).firstElement);

  cublasDestroy(cublas_handle);
  cusolverDnDestroy(solver_handle);

  return 0;
}
