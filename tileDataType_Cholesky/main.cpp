#include <stdlib.h>

#include <iostream>
#include <string>

#include "api_cholesky.h"
#include "miscellaneous_functions.h"

int main() {
  // parameters setup
  // number of rows and cols of matrix A
  int M = 9;
  // number of rows in tiles of A
  int M_tile = 3;

  // matrix creation on CPU
  Matrix<double> A(M, M, M_tile, M_tile);

  // fill in original matrices
  setPDMatrix(A.firstElement, A.rowsMatrix);

  std::cout << "********* matrix information ********* " << std::endl;
  std::cout << "number of rows in A: " << A.rowsMatrix << std::endl;
  std::cout << "tile size: " << A.tilesInRow << std::endl;
  std::cout << "number of tiles in along each dimension: " << A.tilesInRow << std::endl;
  std::cout << "initiliazed SPD matrix A: " << std::endl;
  print_matrix(A.firstElement, M, M, M_tile, M_tile);
  
  // ********************************
  // Cholesky factorization
  const int numTiles = A.tilesInRow;
  int info_cholesky[numTiles];  
  double alpha_trsm =1.0;
  double alpha =-1.0;
  double beta =1.0;
    
  for(int k =0; k < numTiles; k++)
  {
  	std::cout << "value of k="<< k <<std::endl;
  	dpotrf_tile(A(k, k), info_cholesky + k -1 );
  	std::cout << "DPOTRF on A(" << k << ", " << k << ")" <<std::endl;
  	
  	for(int m=k+1; m < numTiles; m++)
  	{
  		//dtrsm_tile(const Tile<T> tileA, Tile<T> tileB, T *alpha) 
  		std::cout << "\t DTRSM, m=" << m <<std::endl;
  		dtrsm_tile( A(k, k), A(m, k), &alpha_trsm);  		
  	}
  		
  	for(int n = k+1; n < numTiles; n++)
  	{
  		std::cout << "\t DSYRK, n=" << n <<std::endl;
  		//dsyrk_tile(const Tile<T> tileA, Tile<T> tileC, T *alpha, T *beta)
  		dsyrk_tile( A(n, k), A(n, n), &alpha, &beta );
  		
  		for(int m=n+1; m<numTiles; m++)
  		{
  			std::cout << "\t\t GEMM, m=" << m <<std::endl;
  			// dgemm_tile(const Tile<T> tileA, const Tile<T> tileB, Tile<T> tileC, T *alpha, T *beta)
  			dgemm_tile(A(m, k), A(n, k), A(m, n), &alpha, &beta);
  		}
  	}
  	std::cout << "------------------------"<<std::endl;
  }
  
  extractTriangular(A.firstElement, A.rowsMatrix);
  std::cout << "********* Cholesky factor L ********* " << std::endl;
  print_matrix(A.firstElement, M, M, M_tile, M_tile);
  
  // ****************************************
 
  // free dynamic allocated memory
  free(A(0, 0).firstElement);

  return 0;
}
