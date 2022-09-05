#include <stdlib.h>

#include <iostream>
#include <string>
#include <vector>

template <typename T>
struct Tile {
  // rows in tile
  int rowsTile;
  // columns in tile
  int colsTile;
  int ld;

  __device__ __host__ T* firstElement;

  __device__ __host__ const T& operator()(int row, int col) const noexcept {
    return &firstElement[ld * col + row];
  }
};

template <typename T>
struct Matrix {
  // how many rows the matrix has
  int rowsMatrix;
  // how many cols the matrix has
  int colsMatrix;
  // how many tiles fit along row direction
  int tilesInRow;
  // how many tiles fit along column direction
  int tilesInCol;
  // ld is the leading dimension; we assume column major indexing
  int ld;
  // this array stores pointers to first element of each tile
  // allocated on the GPU
  std::vector<Tile<T>> tiles;

  int rowsTile;
  int colsTile;

  // constructor
  // we pass a pointer to the first element of the matrix; we assume we're
  // given a GPU-allocated matrix
  Matrix(const T* h_A, const int numRowMatrix, const int numColMatrix,
         const int numRowTile, const int numColTile) {
    // assign value to matrix parameters
    rowsMatrix = numRowMatrix;
    colsMatrix = numColMatrix;
    rowsTile = colsTile;
    colsTile = numColTile;
    ld = numRowMatrix;

    // number of TILES of dimension (numRowTile x numColTile) along row
    // dimension
    int totalTileRow = int(numRowMatrix / numRowTile);
    // if number of ROWS in the tiles that cannot be of dimension (numRowTile x
    // numColTile)
    const int remainingTileRow = numRowMatrix % numRowTile;
    // number of TILES of dimension (numRowTile x numColTile) along col
    // dimension
    int totalTileCol = int(numColMatrix / numColTile);
    // if number of ROWS in the tiles that cannot be of dimension (numRowTile x
    // numColTile)
    const int remainingTileCol = numColMatrix % numColTile;

    if (remainingTileRow != 0) totalTileRow += 1;
    if (remainingTileCol != 0) totalTileCol += 1;

    tilesInRow = totalTileRow;
    tilesInCol = totalTileCol;

    // pointer to memory on the GPU
    T* d_A;
    // allocate once memory for original matrix on the GPU
    cudaMalloc(&d_A, rowsMatrix * colsMatrix * sizeof(T));
    // copy matrix allocated on the CPU to the GPU
    cudaMemcpy(d_A, h_A, rowsMatrix * colsMatrix * sizeof(T),
               cudaMemcpyHostToDevice);

    int counter_rows = 0;
    int counter_cols = 0;
    int temp_numRowTile = 0;
    int temp_numColTile = 0;

    for (int j = 0; j < totalTileCol; j++) {
      temp_numColTile = numColTile;
      if ((remainingTileCol != 0) && (counter_cols == totalTileCol - 1)) {
        temp_numColTile = remainingTileCol;
      }

      Tile<T> t;
      for (int i = 0; i < totalTileRow; i++) {
        temp_numRowTile = numRowTile;

        if ((remainingTileRow != 0) && (counter_rows == totalTileRow - 1)) {
          temp_numRowTile = remainingTileRow;
        }

        t.rowsTile = temp_numRowTile;
        t.colsTile = temp_numColTile;
        t.ld = numRowMatrix;
        t.firstElement = &d_A[numColTile * ld * j + numRowTile * i];
        tiles.push_back(t);

        counter_rows += 1;
      }
      counter_rows = 0;
      counter_cols += 1;
    }
  }

  //__device__ __host__
  const Tile<T>& operator()(int i, int j) const noexcept {
    // it is assumed the indexing starts at 0 and colum-major indexing
    return tiles[j * tilesInRow + i];
  }
};
