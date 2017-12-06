#ifndef KDTTENSOR_H_
#define KDTTENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cblas.h>
#include <omp.h>

/**
This is a header file that contains defiition for:
	1) KTENSOR
	2) TENSOR
	3) Hadamard Product - Hadamard()
	4) Multi_KR() - function do to multiple KR products
	5) printM() - function to print a matrix in column major odering
	6) compareM() - function for comparing two matrices using a certain tolerance
	7) TransposeM()
	8) MHada()

Move function to its own c file with MTTKRP
make inputs to KR just a ktensor struct

NOTES and QUESTIONS:
-should we make functions that act like constructors that take points to 
	various structs and fill them with given argument values;
	EXAMPLE: ktensor* ktensor_const(double ** factors, int * dims, int nmodes, int rank){//FILL VALUES}

*/


/**
KTENSOR definition 
Should Include: 
	1) Array of Factor Matrices 	(double **)
	2) Array of Dimensions 		(int *)
	3) Number of Modes 		(int)
	4) Rank  			(int)
	5) Weights or Lambda values 
*/

typedef struct{

	double 	**	factors;
	int 	* 	dims;
	int 		nmodes;
	int 		rank;
	int 		dims_product;
	double 	* 	lambdas;

}ktensor;

/**
TENSOR definition
Should Include:
	1) Data			(double *)
	2) Dims			(int *)
	3) Number of Modes	(int)

	- the Data is stored column wise in a the first mode matricization of the tensor.
*/

typedef struct{

	double	* 	data;
	int 	* 	dims;
	int		nmodes;
	int 		dims_product;

}tensor;

/**
	For processing command line inputs	
*/
typedef struct{

	int 	rank;
	int 	nmodes;
	double 	noise;
	int 	num_threads;
	int	max_iters;
	double  tolerance;
	int * 	dims;

}tensor_inputs;

/**
	Left or Right direction type
	Used as an input for the MTTKRP 2Step function
*/
typedef enum {left, right} direction;

/**
	Function Prototypes
*/

/**
	Utility fuctions
	
	printM_ColMajor() 		- prints a marix in column major order
	printM_RowMajor() 		- prints a Matrix in row-major order
	print_Ktensor_RowMajor() 	- prints a Ktensor object
	TransposeM()			- transposes the matrix A into A_T
	CompareM()			- compares two matrices of the same size element by element
	compute_Iteration_Space()	- divides up an iteration space amongst threads
	Gen_Tensor()			- generates a rank R tensor
	clean_Up_Gen_Tensor()		- cleans up a tensor and ktensor
	print_dgemm_inputs()		- prints inputs to a cblas_dgemm call
	print_dgemv_inputs()		- prints inputs to a cblas_dgemv call
	Full_nMode_Matricization_RowMajor()	- creates a matricization of a tensor from a tensor model
*/
void printM_ColMajor( double * M, int num_cols, int num_rows );
void printM_RowMajor(double* M, int num_cols, int num_rows);
void print_Ktensor_RowMajor( ktensor * Y );
void TransposeM( double * A, double * A_T, int rowsA, int colsA );
int CompareM( double * A, double * B, int nRows, int nCols, double eps );
void compute_Iteration_Space( int thread_id, int num_threads, int iter_space ,int * start, int * end );
void Gen_Tensor( ktensor * Y, tensor * T, int R, int N, int * D, double noise );
void clean_Up_Gen_Tensor( ktensor * Y, tensor * T );
void print_tensor( tensor * T, int show_data );
void print_dgemm_inputs( CBLAS_ORDER dgemm_layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n , int k, double alpha, int strideA, int strideB, double beta, int strideC );
void print_dgemv_inputs( CBLAS_ORDER dgemv_layout, CBLAS_TRANSPOSE transA, int m, int n, double alpha, int T_offset, int tensor_stride, int A_offset, int A_stride, double beta, int output_col_stride, int output_stride);
void Full_nMode_Matricization_RowMajor( tensor * T, ktensor * Y, int n );

/**
	Sequential KRP functions
	
	Multi_KR_RowMajor()		- Does the rev-KRP of all factor matrices of Y, skipping the nth mode, has reuse
	KR_RowMajor()			- Does the rev-KRP of 2 factor matrices
	update_Partial_Hadamards()	- Updates partial hadamards for Multi_KR_RowMajor()
	
*/
void Multi_KR_RowMajor( ktensor * Y , double * C, int n );
void KR_RowMajor( ktensor * Y, double * C, int n );
void update_Partial_Hadamards( ktensor * Y, int * indexers, double * partial_Hadamards, double ** reordered_Factors );

/**
	Parallel KRP functions
	
	parallel_Multi_KR_RowMajor()		- computes the pieces of a KRP of all given factor matrices in Y
	parallel_KR_RowMajor()			- handles the Y->nmodes == 2 case fo above
	parallel_update_Partial_Hadamards()	- updates the Partial_Hadamards for the above
	wrapper_Parallel_Multi_revKRP()		- wrapper function for doing a parallel revKRP, uses the 3 above functions
*/
void parallel_Multi_KR_RowMajor( ktensor * Y, double * C, int start, int end);
void parallel_KR_RowMajor( ktensor * Y, double * C, int start, int end );
void parallel_update_Partial_Hadamards( ktensor * Y, int * indexers, double * partial_Hadamards );
void wrapper_Parallel_Multi_revKRP( ktensor * Y, int num_threads, double * KRP );

/**
	CP_ALS functions

	MTTKRP_rowMajor()			- performs a slicing(no permuting needed) MTTKRP in row major
	Upper_Hadamard_RowMajor() 		- Hadamard product of upper part of symmetric matrices in row major order
	normalize_Factor_Matrix_RowMajor()	- normalizes the nth factor matrix for a row major ktensor
	normalize_Ktensor_RowMajor()		- normalizes all the factor matrices for a row major ktensor
	do_SYRKs_RowMajor()			- does the SYRKS of all factor matrices in a given Y and stores them
	approximation_Error()			- comptues the error of a given CP model against the original tensor
	MHada_RowMajor()			- does the hadamard product of upper triangular matrices
*/
void MTTKRP_RowMajor( tensor * T, double * K, double * C, int rank, int n);
void Upper_Hadamard_RowMajor( int nRows, int nCols, double * A, double * B, double * C );
void normalize_Factor_Matrix_RowMajor( ktensor * Y, int n );
void normalize_Ktensor_RowMajor( ktensor * Y );
void do_SYRKs_RowMajor( ktensor * Y, double ** SYRKs );
double approximation_Error( double * X, double * KR, ktensor * Y, int n );
double CP_ALS_efficient_error_computation( ktensor * Y, int n, double * MTTKRP, double * V, double * S, double tensor_norm );
double CP_ALS_naive_error_computation( tensor * T, ktensor * Y, double * X, double * KRP, int n );
void MHada_RowMajor( ktensor * Y, double ** SYRKs, double * V, int n );

void reorder_Factors( ktensor * Y, double ** reordered_Factors, int * reordered_Dims, int n );
void reorder_Ktensor( ktensor * Y, ktensor * nY, int n );
void compute_KRP_Indices( int j, ktensor * Y, int * indeces );

void update_Partial_Hadamards_2( ktensor * Y, int * indexers, double * partial_Hadamards );

/**
	Input and object validity functions

	dims_check()		- checks an array of dimensions with the lenght to verify the validity
	process_inputs()	- processes general inputs for generating a rank R tensor
*/
void dims_check( int * dims, int length);
void process_inputs( int argc, char *argv[], tensor_inputs * inputs );
void destroy_inputs( tensor_inputs * inputs );							// Parallel MTTKRP function

// Functions for manipulating and manageing ktensor and tensor objects
void LR_Ktensor_Reordering_newY( ktensor * Y, ktensor * nY, int n, direction D );
void LR_Ktensor_Reordering_existingY( ktensor * Y, ktensor * nY, int n, direction D );
void destruct_Ktensor( ktensor * Y, int clear_factors );
void destruct_Tensor( tensor * T );
void LR_tensor_Reduction( tensor * T, tensor * nT, int n, direction D );
void tensor_data_swap( tensor * t1, tensor * t2 );
void ktensor_copy_constructor( ktensor * Y, ktensor * nY );
direction opposite_direction( direction D );
void remove_mode_Ktensor( ktensor * Y, int n );

//mkl unavailable functions
void vdMul(int , double *, double *, double *);

#endif  // KDTTENSOR_H_
