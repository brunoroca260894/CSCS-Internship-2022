#include <stdlib.h>

#include <iostream>
#include <string>

#include "tile_type_GPU.h"

template <typename T>
cusolverStatus_t cublas_dpotrf_tile(cusolverDnHandle_t solver_handle,
                                    Tile<T> tileA, int *h_devInfo) {
  cublasFillMode_t UPLO = CUBLAS_FILL_MODE_LOWER;
  int N = tileA.rowsTile;
  int LDA = tileA.ld;

  int work_size;
  int *devInfo;
  cudaMalloc(&devInfo, sizeof(int));

  cusolverDnDpotrf_bufferSize(solver_handle, UPLO, N, tileA.firstElement, LDA,
                              &work_size);

  double *work;
  cudaMalloc(&work, work_size * sizeof(double));

  // actual Cholesky factorization
  cusolverStatus_t cholStatus =
      cusolverDnDpotrf(solver_handle, UPLO, N, tileA.firstElement, LDA, work,
                       work_size, devInfo);

  cudaMemcpy(h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

  // cudaFree(devInfo);
  // cudaFree(work);

  return cholStatus;
}

template <typename T>
cublasStatus_t cublas_dgemm_tile(cublasHandle_t handle, const Tile<T> tileA,
                                 const Tile<T> tileB, Tile<T> tileC, T *alpha,
                                 T *beta) {
  cublasOperation_t TRANSA = CUBLAS_OP_N;
  cublasOperation_t TRANSB = CUBLAS_OP_T;  // must be CUBLAS_OP_T for cholesky
  int M = tileA.rowsTile;
  int N = tileB.colsTile;
  int K = tileA.colsTile;  // = tB1.rowsTile
  int LDA = tileA.ld;
  int LDB = tileB.ld;
  int LDC = tileC.ld;

  cublasStatus_t gemmStatus =
      cublasDgemm(handle, TRANSA, TRANSB, M, N, K, alpha, tileA.firstElement,
                  LDA, tileB.firstElement, LDB, beta, tileC.firstElement, LDC);

  return gemmStatus;
}

template <typename T>
cublasStatus_t cublas_dtrsm_tile(cublasHandle_t handle, const Tile<T> tileA,
                                 Tile<T> tileB, T *alpha) {
  cublasSideMode_t SIDE = CUBLAS_SIDE_RIGHT;
  cublasFillMode_t UPLO = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t TRANSA = CUBLAS_OP_T;
  cublasDiagType_t DIAG = CUBLAS_DIAG_NON_UNIT;
  int M = tileB.rowsTile;  // rows in tileB
  int N = tileB.colsTile;  // cols in tileB
  int LDA = tileA.ld;
  int LDB = tileB.ld;

  cublasStatus_t trsmStatus =
      cublasDtrsm(handle, SIDE, UPLO, TRANSA, DIAG, M, N, alpha,
                  tileA.firstElement, LDA, tileB.firstElement, LDB);

  return trsmStatus;
}

template <typename T>
cublasStatus_t cublas_dsyrk_tile(cublasHandle_t handle, const Tile<T> tileA,
                                 Tile<T> tileC, T *alpha, T *beta) {
  cublasFillMode_t UPLO = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t TRANS = CUBLAS_OP_N;
  int N = tileC.colsTile;
  int K = tileA.colsTile;  // = tB1.rowsTile
  int LDA = tileA.ld;
  int LDC = tileC.ld;

  cublasStatus_t syrkStatus =
      cublasDsyrk(handle, UPLO, TRANS, N, K, alpha, tileA.firstElement, LDA,
                  beta, tileC.firstElement, LDC);

  return syrkStatus;
}
