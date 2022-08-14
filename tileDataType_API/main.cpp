#include <stdlib.h>

#include <iostream>
#include <string>

#include "api_cholesky.h"
#include "miscellaneous_functions.h"

int main() {
  // parameters setup
  // number of rows of matrix A
  int M = 6;
  // number of cols of matrix B
  int N = 6;
  // number of cols of matrix A (or rows of matrix B)
  int K = 8;
  // number of rows in tiles of A
  int M_tile = 2;
  // number of cols in tile of B
  int N_tile = 2;
  // number of cols in tile of A (or rows in tile of V)
  int K_tile = 3;
  // how to fill the matrix
  char transAFill = 'N';

  // matrix creation on CPU
  Matrix<double> A(M, K, M_tile, K_tile);
  Matrix<double> B(K, N, K_tile, N_tile);
  Matrix<double> C(M, N, M_tile, N_tile);

  // fill in original matrices
  fillin_matrix(A.firstElement, A.rowsMatrix, A.colsMatrix, &transAFill);
  fillin_matrix(B.firstElement, B.rowsMatrix, B.colsMatrix, &transAFill);
  fillin_matrix_zero(C.firstElement, C.rowsMatrix, C.colsMatrix);

  std::cout << " *********DGEMM results********* " << std::endl;
  std::cout << "initiliazed matrix A: " << std::endl;
  print_matrix(A.firstElement, M, K, M_tile, K_tile);

  std::cout << "initiliazed matrix B: " << std::endl;
  print_matrix(B.firstElement, K, N, K_tile, N_tile);

  std::cout << "initiliazed matrix C: " << std::endl;
  print_matrix(C.firstElement, M, N, M_tile, N_tile);

  // ********************************
  // tiled-dgemm multiplication
  // A is an (M, K) matrix and B is a (K, N) matrix
  char TRANSA = 'N';
  char TRANSB = 'N';
  // total tiles along row direction of B
  int M_grid = A.tilesInRow;
  // total tiles along col direction of B
  int N_grid = B.tilesInCol;
  // total tiles along col and row directions of A and B, respectively
  int K_grid = A.tilesInCol;
  double ALPHA = 1.0;
  // we set beta =1 since we want to sum the product tileA*tileA for each tile
  // in C
  double BETA = 1.0;

  // actual tiled-gemm computation
  for (int i = 0; i < M_grid; i++) {
    for (int j = 0; j < N_grid; j++) {
      for (int k = 0; k < K_grid; k++)
        dgemm_tile(A(i + 1, k + 1), B(k + 1, j + 1), C(i + 1, j + 1), &TRANSA,
                   &TRANSB, &ALPHA, &BETA);
    }
  }

  std::cout << "the product matrix C = AB is: " << std::endl;
  print_matrix(C.firstElement, M, N, M_tile, N_tile);

  // ********************************
  /// tiled-DSYRK function
  char UPLO = 'U';
  TRANSA = 'N';
  TRANSB = 'N';
  ALPHA = 1;
  BETA = 1;
  transAFill = 'N';
  // how matrix B2 is filled
  char transBFill = 'T';

  // matrix A2 defintion
  // rows in A2
  N = 6;
  // cols in A2
  K = 8;
  // rows in tiles of A2
  N_tile = 2;
  // cols in tiles of A2
  K_tile = 3;

  // matrix creation on CPU
  Matrix<double> A2(N, K, N_tile, K_tile);
  Matrix<double> B2(K, N, K_tile, N_tile);
  Matrix<double> C2(N, N, N_tile, N_tile);

  fillin_matrix(A2.firstElement, A2.rowsMatrix, A2.colsMatrix, &transAFill);
  fillin_matrix(B2.firstElement, B2.rowsMatrix, B2.colsMatrix, &transBFill);
  fillin_matrix_zero(C2.firstElement, C2.rowsMatrix, C2.colsMatrix);

  std::cout << " *********DSYRK results********* " << std::endl;
  std::cout << "initiliazed matrix A2: " << std::endl;
  print_matrix(A2.firstElement, N, K, N_tile, K_tile);
  std::cout << "initiliazed matrix B2: " << std::endl;
  print_matrix(B2.firstElement, K, N, K_tile, N_tile);
  std::cout << "initiliazed matrix C2: " << std::endl;
  print_matrix(C2.firstElement, N, N, N_tile, N_tile);

  // rows in each tile of C
  N_grid = A.tilesInRow;
  // cols in each tile of A
  K_grid = A.tilesInCol;

  for (int i = 0; i < N_grid; i++) {
    for (int j = 0; j < N_grid; j++) {
      if (j == i)  // diagonal elements
      {
        for (int k = 0; k < K_grid; k++) {
          dsyrk_tile(A(i + 1, k + 1), C2(i + 1, j + 1), &UPLO, &ALPHA, &BETA);
        }
      }
      if (j > i)  // off-diagonal elements
      {
        for (int k = 0; k < K_grid; k++) {
          dgemm_tile(A2(i + 1, k + 1), B2(k + 1, j + 1), C2(i + 1, j + 1),
                     &TRANSA, &TRANSB, &ALPHA, &BETA);
        }
      }
    }
  }

  std::cout << "matrix C2 after DSYRK is: " << std::endl;
  print_matrix(C2.firstElement, N, N, N_tile, N_tile);

  // free dynamic allocated memory
  free(A(1, 1).firstElement);
  free(B(1, 1).firstElement);
  free(C(1, 1).firstElement);

  free(A2(1, 1).firstElement);
  free(B2(1, 1).firstElement);
  free(C2(1, 1).firstElement);

  return 0;
}
