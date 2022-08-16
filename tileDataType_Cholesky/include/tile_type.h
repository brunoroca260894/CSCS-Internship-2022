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
  T* firstElement;
  const T& operator()(int row, int col) const noexcept {
    return firstElement[ld * (col - 1) + row - 1];
  }
};

template <typename T>
struct Matrix {
  int rowsMatrix;
  int colsMatrix;
  // how many tiles fit along row direction
  int tilesInRow;
  // how many tiles fit along column direction
  int tilesInCol;
  // ld is the leading dimension, ld is the number of columns
  int ld;
  T* firstElement;  // this could be deleted
  std::vector<Tile<T>> tiles;

  // constructor
  Matrix(const int numRowMatrix, const int numColMatrix, const int numRowTile,
         const int numColTile) {
    // assign value to matrix parameters
    rowsMatrix = numRowMatrix;
    colsMatrix = numColMatrix;
    ld = numRowMatrix;

    // number of TILES of dimension (tileRows x tileCols)
    int totalTileRow = int(numRowMatrix / numRowTile);
    // number of ROWS in the tiles that cannot be of dimension (numRowTile x
    // numColTile)
    const int remainingTileRow = numRowMatrix % numRowTile;
    // number of TILES of dimension (tileRows x tileCols)
    int totalTileCol = int(numColMatrix / numColTile);
    // number of ROWS in the tiles that cannot be of dimension (numRowTile x
    // numColTile)
    const int remainingTileCol = numColMatrix % numColTile;

    if (remainingTileRow != 0) totalTileRow += 1;
    if (remainingTileCol != 0) totalTileCol += 1;

    tilesInRow = totalTileRow;
    tilesInCol = totalTileCol;

    // allocate once memory for original matrix
    T* fullMatrix = (T*)malloc(int(rowsMatrix * colsMatrix) * sizeof(T));
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
        t.firstElement = &fullMatrix[numColTile * ld * j + numRowTile * i];
        tiles.push_back(t);

        counter_rows += 1;
      }
      counter_rows = 0;
      counter_cols += 1;
    }
    // pointer to the first element of the matrix
    firstElement = tiles.front().firstElement;
  }

  const Tile<T>& operator()(int i, int j) const noexcept {
    // it is assumed the indexing starts at 1
    return tiles[(j - 1) * tilesInRow + i - 1];
  }
};
