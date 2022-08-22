#include <chrono>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "api_cholesky.h"
#include "miscellaneous_functions.h"

/*
 * Parallel implementation based on:
 * 1. Dorris, J., Kurzak, J., Luszczek, P., YarKhan, A., & Dongarra, J. (2016, June). Task-based Cholesky decomposition on knights corner
 * using OpenMP. In International Conference on High Performance Computing (pp. 544-562). Springer, Cham.
 * And
 * 2. Ltaief, H., Tomov, S., Nath, R., & Dongarra, J. (2010). Hybrid multicore cholesky factorization with multiple gpu accelerators. IEEE
 * Transaction on Parallel and Distributed Systems, 48.
 */

int main()
{
    // parameters setup
    // number of rows and cols of matrix A
    int M;
    // number of rows in tiles of A
    int M_tile;

    // Cholesky factorization parameters
    int info_cholesky;
    double alpha_trsm = 1.0;
    double alpha = -1.0;
    double beta = 1.0;

    // openmp parameters
    int numThreads;

    // file to save original matrix and its Cholesky factor L
    const char* fileL = "/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/choleskyFactor.txt";
    const char* fileMatrix = "/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/originalMatrix.txt";
    //const char* fileTimeTiledCholesky = "/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/timeTiledCholesky.txt";
    
    const char* fileTimeTiledCholesky[5] = {"/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/timeTiledCholesky_p05.txt",
    "/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/timeTiledCholesky_p10.txt",
    "/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/timeTiledCholesky_p15.txt",
    "/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/timeTiledCholesky_p20.txt",
    "/home/bruno/Documents/internhsip/tileDataType/v03_cholesky/timeTiledCholesky_p25.txt"
    };

    double tileSize[] = {0.05, 0.1, 0.15, 0.2, 0.25};
    
    int matrixSize[] = {1000, 2500, 5000, 7500, 10000, 12500, 15000};
    int threadSize[] = { 1, 2, 4, 8, 12 };
       
    double timeTiledCholesky[int(sizeof(matrixSize) * sizeof(threadSize))];

    for (int i = 0; i < int( sizeof(tileSize)/sizeof(double) ); i++) // iterate over different tile sizes
    {    	
    	std::cout << "****************** TILE VALUE =" << tileSize[i] << "******************" << std::endl;	
        for (int j = 0; j < int( sizeof(matrixSize)/sizeof(int) ); j++) // iterate over different matrix sizes
        {
        	
        	M = matrixSize[j];
            M_tile = int(tileSize[i] * matrixSize[j]);
            std::cout << "* matrix size =" << matrixSize[j] << " *" << std::endl;	
        	std::cout << "* tile size =" << M_tile << " *" << std::endl;
        	
            for (int l = 0; l < int( sizeof(threadSize)/sizeof(int) ); l++) // iterate over different number of threads
            {
            	 
                // matrix creation on CPU
                Matrix<double> A(M, M, M_tile, M_tile);
                // fill in original matrices
                set_SPD_Matrix(A.firstElement, A.rowsMatrix);

                // Cholesky factorization parameters
                int numTiles = A.tilesInRow;                
                                
                // openMP
                numThreads = threadSize[l];
                omp_set_num_threads(numThreads);
                                					
				std::cout << "\t number of threads =" << numThreads << std::endl;          
                
                auto begin = std::chrono::high_resolution_clock::now();

// actual Tiled Cholesky computation
#pragma omp parallel
#pragma omp master
                for (int k = 0; k < numTiles; k++) {
#pragma opm task depend(inout \
                        : A(k, k))
                    {
                        dpotrf_tile(A(k, k), &info_cholesky);
                    }

                    for (int m = k + 1; m < numTiles; m++) {
#pragma omp task depend(in                      \
                        : A(k, k)) depend(inout \
                                          : A(m, k))
                        {
                            dtrsm_tile(A(k, k), A(m, k), &alpha_trsm);
                        }
                    }

                    for (int n = k + 1; n < numTiles; n++) {
#pragma omp task depend(in                      \
                        : A(n, k)) depend(inout \
                                          : A(n, n))
                        {
                            dsyrk_tile(A(n, k), A(n, n), &alpha, &beta);
                        }

                        for (int m = n + 1; m < numTiles; m++) {
#pragma omp task depend(in                               \
                        : A(n, k), A(m, k)) depend(inout \
                                                   : A(m, n))
                            {
                                dgemm_tile(A(m, k), A(n, k), A(m, n), &alpha, &beta);
                            }
                        }
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
                std::cout << "\t total time measured in seconds " << elapsed.count() * 1e-6 << std::endl;
                
                timeTiledCholesky[ int( sizeof(threadSize)/sizeof(int) )*j + l ] = elapsed.count() * 1e-6;                              
                
                // free dynamic allocated memory
    			free(A(0, 0).firstElement);
            }
        }
        save_data(timeTiledCholesky, fileTimeTiledCholesky[i], int( sizeof(matrixSize)/sizeof(int) ), int( sizeof(threadSize)/sizeof(int) ));
        std::cout << "------------ooo000ooo--------------" << std::endl;        
    }   

    return 0;
}
