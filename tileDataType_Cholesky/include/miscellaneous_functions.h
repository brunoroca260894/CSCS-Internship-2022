#include <stdlib.h>
#include <iostream>
#include <string>
#include <iomanip>
#include<fstream>

#define prec_save 8

// print matrix
template <typename T>
void print_matrix(const T* A, const int rowsA, const int colsA,
                  const int rowsTile, const int colsTile) {
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsA; j++) {
      std::cout << std::setw(8) << std::fixed << std::setprecision(prec_save) << A[j * rowsA + i] << ",\t";
    }
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

// create spd matrix
template <typename T>
void setPDMatrix(T* __restrict A, const int N) {

    // --- Initialize random seed
    srand(0);
        
    T* Atemp = (T*)malloc(N * N * sizeof(T));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Atemp[i * N + j] = (T)rand() / (T)RAND_MAX;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) 
            A[i * N + j] = 0.5 * (Atemp[i * N + j] + Atemp[j * N + i]);

    for (int i = 0; i < N; i++) A[i * N + i] = A[i * N + i] + N;
    
    free(Atemp);
}


/************************************/
/* SAVE ARRAY FROM CPU TO FILE */
/************************************/
template <class T>
void saveCPUrealtxt(const T* h_in, const char *filename, const int N) {
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
	    outfile << std::setw(8) << std::fixed << h_in[i+j*N] << "\t";
    	}    	    	
    }           	
    outfile.close();
}

template <typename T>
void extractTriangular(T* A, const int numRowMatrix)
{
	for(int j=0; j < numRowMatrix; j++) 
    {
    	for(int i=0; i < numRowMatrix; i++)
    	{
    		if(i<j) A[j * numRowMatrix + i] = 0; 
        }
    } 
}
