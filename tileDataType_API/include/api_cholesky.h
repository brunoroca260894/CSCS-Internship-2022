#include <stdlib.h>

#include <iostream>
#include <string>

#include "tile_type.h"

extern "C" void dgemm_(const char *TRANSA, const char *TRANSB, const int *M,
                       const int *N, const int *K, double *ALPHA, double *A,
                       const int *LDA, double *B, const int *LDB, double *BETA,
                       double *C, const int *LDC);

extern "C" void dtrsm_(const char *SIDE, const char *UPLO, const char *TRANSA,
                       const char *DIAG, const int *M, const int *N,
                       double *ALPHA, double *A, const int *LDA, double *B,
                       const int *LDB);

extern "C" void dsyrk_(const char *UPLO, const char *TRANS, const int *N,
                       const int *K, double *ALPHA, double *A, const int *LDA,
                       double *BETA, double *C, const int *LDC);

template <typename T>
void dgemm_tile(const Tile<T> tileA, const Tile<T> tileB, Tile<T> tileC,
                const char *TRANSA, const char *TRANSB, T *alpha, T *beta) {
  int M = tileA.rowsTile;
  int N = tileB.colsTile;
  int K = tileA.colsTile;  // = tB1.rowsTile
  int LDA = tileA.ld;
  int LDB = tileB.ld;
  int LDC = tileC.ld;

  dgemm_(TRANSA, TRANSB, &M, &N, &K, alpha, tileA.firstElement, &LDA,
         tileB.firstElement, &LDB, beta, tileC.firstElement, &LDC);
}

template <typename T>
void dsyrk_tile(const Tile<T> tileA, Tile<T> tileC, const char *UPLO, T *alpha,
                T *beta) {
  // char UPLO = 'U';
  char TRANS = 'N';
  int N = tileC.colsTile;
  int K = tileA.colsTile;  // = tB1.rowsTile
  int LDA = tileA.ld;
  int LDC = tileC.ld;

  // dsyrk_(&UPLO, &TRANS, &N, &K, alpha, tileA.firstElement, &LDA, beta,
  // tileC.firstElement, &LDC);
  dsyrk_(UPLO, &TRANS, &N, &K, alpha, tileA.firstElement, &LDA, beta,
         tileC.firstElement, &LDC);
}
