#include<iostream>
#include <stdlib.h> 
#include <string>

/*
 * I assume that we are given the number of tiles or submatrix and their corresponding sizes. 
 * I create a structure that stores each tile's width, height, leading dimension, and 
 * a pointer to the first element of each tile, which is in turn the element at the upper left 
 * part of each pointer. From this, we could first say that we can create a grid of tiles.
 * Moreover, at tile level, we can access each tile element by using the pointer we already have 
 * and the tile width and height.
 * We should keep in mind that elements are stored in row-najor order. 
 */

template <typename T> 
struct Matrix { 
  int width; 
  int height; 
  // ld is the leading dimension, we assume here the leading dimension is the number of columns since
  // in C++ the row-major order is followed
  int ld; 
  T* element; 
}; 

// prototype for tile in general grid
template <typename T> 
Matrix<T> getTile(const struct Matrix<T> A, const int rowGrid, const int colGrid, const int tileWidth, const int tileHeight);

template <typename T> 
T getElement(const struct Matrix<T> A, const int row, const int col);

template <typename T> 
Matrix<T> getTile(const struct Matrix<T> A, const int rowGrid, const int colGrid, const int tileWidth, const int tileHeight)
{
	Matrix<T> tile; 
	tile.width = tileWidth;
	tile.height = tileHeight;
	// ld is the leading dimension, we assume here the leading dimension is the number of columns since
	// in C++ the row-major ordering is followed
	tile.ld = A.ld;
	// pointer to first element in a tile, element at the left upper corner
	tile.element = &A.element[ A.ld * tile.height * rowGrid + tile.width * colGrid ];
	return tile;
}

// To access element at A(row, col)
// we can think of A as a tile, and row and col as the number of rows and cols in each tile
template <typename T> 
T getElement(const struct Matrix<T> A, const int row, const int col)
{
	return A.element[ row * A.ld + col ];
}

// print matrix
template <typename T> 
void print_matrix(const T* A, const int Awidth, const int Aheight, const int tileWidth, const int tileHeight) 
{
    for(int i = 0; i < Aheight; i++)
    {
    	if( (0 ==i%tileHeight) && (0 != i) ){	
    		for( int k = 0; k < Awidth + int(Awidth/tileWidth) -1 ; k++) 
    			std::cout << "*" << "\t";     
    		std::cout << std::endl;		   		
    	}     
    	    	
        for(int j = 0; j < Awidth; j++){
        	if( (0 ==j%tileWidth) && (0 != j) )	
				std::cout << "*\t" ;    	   		
            std::cout << A[i * Awidth + j] << "\t";
        }
         
        // new line after printing a entire row
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/********/
/* MAIN */
/********/
int main()
{
	// parameters setup	
	const int tileWidth = 2;
	const int tileHeight = 3;
	// number of tiles in original matrix, we create a grid of 5*3 tiles of size tileHeight*tileWidth each
	const int numberTilesCol = 5; //number of tiles in horizontal axis
    const int numberTilesRow = 3; //number of tiles in vertical axis
	
	//general matrix creation on CPU
    Matrix<float> A;
    A.width =int(numberTilesCol*tileWidth);
    A.height =int(numberTilesRow*tileHeight);
    A.ld = A.width;
    float* h_A = (float*)malloc(A.width*A.height*sizeof(float));
    A.element = h_A;
    
    // fill in matrix A
    for(int i = 0; i< int(A.width*A.height); i++){
        *(h_A+i) = i+1;
       }
    
    // print matrix in tiles
    std::cout<<"original matrix printed in blocks " << std::endl;
	print_matrix<float> (h_A, A.width, A.height, tileWidth, tileHeight);
	
	// print first element in each tile
	// array of struct Matrix<T> variables
	struct Matrix<float>* tile_heads;
	tile_heads = (struct Matrix<float>*)malloc(numberTilesCol*numberTilesRow*sizeof(struct Matrix<float>));
	
	for(int i=0; i<numberTilesRow; i++)
	{
		for(int j=0; j<numberTilesCol; j++)
		{			
			tile_heads[i*numberTilesCol+ j] = getTile<float>(A, i, j, tileWidth, tileHeight);
			std::cout<<"first element in tile at grid ("<< i<< ", "<<j<< "): "<< *(tile_heads[i*numberTilesCol+ j].element) << std::endl;
		}	
	}			
	
	//std::cout<< "size of struct Matrix: " << sizeof(struct Matrix<float>) << " bytes" << std::endl;
	//std::cout<< "size of array of struct Matrix: " << sizeof(tile_heads) << " bytes" << std::endl;
	
	// create indices for tiles. 
	// we realize that we can access each element in a given tile by using the pointer to the beginning of the tile 
	// and then observe that locally, each tile has 'tileWidth' columns and 'tileHeight' rows	
	// example, let us print all elements in tile at grid point (1,3)
	std::cout<< "elements in tile at grid point (1,3): "<< std::endl;
	for(int i=0; i<tileHeight; i++)
	{
		for(int j=0; j<tileWidth; j++)
		{
			std::cout<< getElement(tile_heads[8], i, j) << "\t";				
			//std::cout<< *(tile_heads[5].element + tile_heads[5].ld*i + j) << " ";
		}	
		std::cout << std::endl;
	}	
	
	free(h_A);
	free(tile_heads);
	
 	return 0;
}
