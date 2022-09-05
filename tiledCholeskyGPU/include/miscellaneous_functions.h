#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#define prec_save 6

/************************************/
/* PRINT MATRIX TO CONSOLE */
/************************************/
template <typename T>
void print_matrix(const T* A, const int rowsA, const int colsA) {
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsA; j++) {
      std::cout << std::setw(prec_save) << std::fixed
                << std::setprecision(prec_save) << A[j * rowsA + i] << ",\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/************************************/
/* PRINT A TILE TO CONSOLE */
/************************************/
template <typename T>
void print_tile(const Tile<T> t) {
  for (int i = 0; i < t.rowsTile; i++) {
    for (int j = 0; j < t.colsTile; j++) {
      std::cout << t(i + 1, j + 1) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/************************************/
/* FILL MATRIX WITH A SEQUENCE */
/************************************/
template <typename T>
void fillin_matrix(T* A, const int numRowMatrix, const int numColMatrix,
                   const char* trans) {
  int counter = 0;
  if ('N' == *trans) {
    for (int j = 0; j < numColMatrix; j++) {
      for (int i = 0; i < numRowMatrix; i++) {
        counter += 1;
        A[j * numRowMatrix + i] = counter;
      }
    }
  } else {
    for (int i = 0; i < numRowMatrix; i++) {
      for (int j = 0; j < numColMatrix; j++) {
        counter += 1;
        A[i + j * numRowMatrix] = counter;
      }
    }
  }
}

/************************************/
/* FILL MATRIX WITH 0 */
/************************************/
template <typename T>
void fillin_matrix_zero(T* A, const int numRowMatrix, const int numColMatrix) {
  for (int j = 0; j < numColMatrix; j++) {
    for (int i = 0; i < numRowMatrix; i++) A[j * numRowMatrix + i] = 0;
  }
}

/************************************/
/* CREATE A SPD MATRIX */
/************************************/
template <typename T>
void set_SPD_Matrix(T* __restrict A, const int N) {
  // --- Initialize random seed to have same sequence
  srand(0);
  // srand(time(NULL));

  T* Atemp = (T*)malloc(N * N * sizeof(T));

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) Atemp[i * N + j] = (T)rand() / (T)RAND_MAX;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A[i * N + j] = 0.5 * (Atemp[i * N + j] + Atemp[j * N + i]);

  for (int i = 0; i < N; i++) A[i * N + i] = A[i * N + i] + N;

  free(Atemp);
}

/************************************/
/* EXTRACT LOWER TRIANGULAR MATRIX */
/************************************/
template <typename T>
void extract_triangular(T* A, const int numRowMatrix) {
  for (int j = 0; j < numRowMatrix; j++) {
    for (int i = 0; i < numRowMatrix; i++) {
      if (i < j) A[j * numRowMatrix + i] = 0;
    }
  }
}

/************************************/
/* SAVE ARRAY FROM CPU TO FILE */
/************************************/
template <class T>
void save_matrix(const T* A, const int N, const char* filename) {
  std::ofstream outfile;
  outfile.open(filename);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      outfile << std::setw(prec_save) << std::fixed << A[i + j * N] << "\t";
    }
    outfile << "\n";
  }
  outfile.close();
}

/************************************/
/* SAVE MEASURED TIME INTO ARRAY */
/************************************/
template <class T>
void save_data(const T* A, const char* filename, const int matrixSize,
               const int numThreads) {
  std::ofstream outfile;
  outfile.open(filename);
  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < numThreads; j++) {
      outfile << std::setw(prec_save) << std::fixed << A[i * numThreads + j]
              << "\t";
    }
    outfile << "\n";
  }
  outfile.close();
}
