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

// Cholesky factorization
extern "C" void dpotrf_(const char *UPLO, const int *N, double *A, const int *LDA, int *INFO);                                    

// actual implementations

template <typename T>
void dpotrf_tile(Tile<T> tileA, int *INFO) {	
  char UPLO ='L';
  int N = tileA.rowsTile;
  int LDA = tileA.ld;
  
  dpotrf_(&UPLO, &N, tileA.firstElement, &LDA, INFO); 
}

template <typename T>
void dgemm_tile(const Tile<T> tileA, const Tile<T> tileB, Tile<T> tileC, T *alpha, T *beta) {
  char TRANSA = 'N';
  char TRANSB = 'T';
  int M = tileA.rowsTile;
  int N = tileB.colsTile;
  int K = tileA.colsTile;  // = tB1.rowsTile
  int LDA = tileA.ld;
  int LDB = tileB.ld;
  int LDC = tileC.ld;

  dgemm_(&TRANSA, &TRANSB, &M, &N, &K, alpha, tileA.firstElement, &LDA,
         tileB.firstElement, &LDB, beta, tileC.firstElement, &LDC);
}

template <typename T>
void dtrsm_tile(const Tile<T> tileA, Tile<T> tileB, T* alpha) {
  char SIDE ='R';
  char UPLO ='L';                       
  char TRANSA = 'T';
  char DIAG = 'N';
  int M = tileB.rowsTile; // rows in tileB  
  int N = tileB.colsTile; // cols in tileB  
  int LDA = tileA.ld;
  int LDB = tileB.ld;
  
  dtrsm_(&SIDE, &UPLO, &TRANSA, &DIAG, &M, &N, alpha, tileA.firstElement, &LDA, tileB.firstElement, &LDB);  
}

template <typename T>
void dsyrk_tile(const Tile<T> tileA, Tile<T> tileC, T *alpha, T *beta) {
  char UPLO = 'L';
  char TRANS = 'N';
  int N = tileC.colsTile;
  int K = tileA.colsTile;  // = tB1.rowsTile
  int LDA = tileA.ld;
  int LDC = tileC.ld;

  dsyrk_(&UPLO, &TRANS, &N, &K, alpha, tileA.firstElement, &LDA, beta,
         tileC.firstElement, &LDC);
}
