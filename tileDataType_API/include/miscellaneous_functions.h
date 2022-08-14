#include <stdlib.h>

#include <iostream>
#include <string>

// print matrix
template <typename T>
void print_matrix(const T* A, const int rowsA, const int colsA,
                  const int rowsTile, const int colsTile) {
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsA; j++) {
      // if( (0 ==j%colsTile) && (0 != j) ) std::cout << "*\t";
      std::cout << A[j * rowsA + i] << "\t";
    }
    /*
    if( (0 ==(i+1)%rowsTile) && (0 != i) ){
            std::cout << std::endl;
            for( int k = 0; k < colsA + int(colsA/colsTile) -1  ; k++) std::cout
    << "*\t";
    }
    */
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// print tile
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

// fill in matrix
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

// fill in matrix with zeros
template <typename T>
void fillin_matrix_zero(T* A, const int numRowMatrix, const int numColMatrix) {
  for (int j = 0; j < numColMatrix; j++) {
    for (int i = 0; i < numRowMatrix; i++) A[j * numRowMatrix + i] = 0;
  }
}
