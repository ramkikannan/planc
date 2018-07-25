/* Copyright Koby Hayashi 2018 */

#ifndef DIMTREE_DDTTENSOR_HPP_
#define DIMTREE_DDTTENSOR_HPP_

#include "dimtree/ddttensor.h"

/*
  Element wise multiplication between two vectors
*/

void vdMul(long int n, double *a, double *b, double *c) {
  for (long int i = 0; i < n; i++) c[i] = a[i] * b[i];
}

/*
  Prints a matrix in column major order
*/
void printM_ColMajor(double *M, long int num_cols, long int num_rows) {
  long int i, j;

  printf("\n");

  for (j = 0; j < num_rows; j++) {
    for (i = 0; i < num_cols; i++) {
      if (i == 0) {
        printf(",%0.10lf", M[i * num_rows + j]);
      } else {
        printf(",%0.10lf", M[i * num_rows + j]);
      }
    }
    printf("\n");
  }
}

/*
  Prints a matrix in row major order
*/
void printM_RowMajor(double *M, long int num_cols, long int num_rows) {
  long int i, j;

  printf("\n");

  for (j = 0; j < num_rows; j++) {
    for (i = 0; i < num_cols; i++) {
      if (i == 0) {
        printf(",%0.10lf", M[j * num_cols + i]);
      } else {
        printf(",%0.10lf", M[j * num_cols + i]);
      }
    }
    printf("\n");
  }
}

void print_Ktensor_RowMajor(ktensor *Y) {
  long int i;

  printf("=========Ktensor========\n");
  printf("Rank = %d\n", Y->rank);
  printf("Nmodes = %d\n", Y->nmodes);
  printf("Dims = [");
  for (i = 0; i < Y->nmodes; i++) {
    if (i != Y->nmodes - 1)
      printf("%d,", Y->dims[i]);
    else
      printf("%d]\n", Y->dims[i]);
  }
  printf("Num_Eles = %d\n", Y->dims_product);
  printf("Lambdas = [");
  for (i = 0; i < Y->rank; i++) {
    if (i != Y->rank - 1)
      printf("%.5lf,", Y->lambdas[i]);
    else
      printf("%.5lf]\n", Y->lambdas[i]);
  }
  for (i = 0; i < Y->nmodes; i++) {
    printf("Factor[%d]", i);
    printM_RowMajor(Y->factors[i], Y->rank, Y->dims[i]);
  }
  printf("========================\n");
}

/*
  Prints the first mode, ColMajor matricization of a tensor struct
  1) T, pointer to the tensor to print
  2) show_data, flag saying if you want T->data to be printed
*/
void print_tensor(tensor *T, long int show_data) {
  long int i;
  printf("==========Tensor========\n");
  printf("T->nmodes = %d\n", T->nmodes);
  printf("T->dims_product = %d\n", T->dims_product);
  printf("T->dims[");
  for (i = 0; i < T->nmodes; i++) {
    printf("%d,", T->dims[i]);
  }
  printf("]\n");

  if (show_data == 1) {
    printf("T->data:");
    printM_ColMajor(T->data, T->dims[0], T->dims_product / T->dims[0]);
  }
  printf("========================\n");
}

/*
  Prints inputs to a cblas_dgemm function call.
*/
void print_dgemm_inputs(CBLAS_ORDER dgemm_layout, CBLAS_TRANSPOSE transA,
                        CBLAS_TRANSPOSE transB, long int m, long int n,
                        long int k, double alpha, long int strideA,
                        long int strideB, double beta, long int strideC) {
  printf("-------------------------------\n");
  printf("dgemm_inputs :: \n");

  if (dgemm_layout == CblasColMajor) {
    printf("ColMajor\n");
  } else {
    printf("RowMajor\n");
  }

  if (transA == CblasNoTrans) {
    printf("NoTrans\n");
  } else {
    printf("Trans\n");
  }

  if (transB == CblasNoTrans) {
    printf("NoTrans\n");
  } else {
    printf("Trans\n");
  }

  printf("m = %d\n", m);
  printf("n = %d\n", n);
  printf("k = %d\n", k);
  printf("alpha = %lf\n", alpha);
  printf("strideA = %d\n", strideA);
  printf("strideB = %d\n", strideB);
  printf("beta = %lf\n", beta);
  printf("strideC = %d\n", strideC);
  printf("-------------------------------\n");
}

/*
  Prints inputs to a cblas_dgemv function call.
*/
void print_dgemv_inputs(CBLAS_ORDER dgemv_layout, CBLAS_TRANSPOSE transA,
                        long int m, long int n, double alpha, long int T_offset,
                        long int tensor_stride, long int A_offset,
                        long int A_stride, double beta,
                        long int output_col_stride, long int output_stride) {
  printf("-------------------------------\n");
  printf("dgemv_inputs :: \n");

  if (dgemv_layout == CblasColMajor) {
    printf("ColMajor\n");
  } else {
    printf("RowMajor\n");
  }

  if (transA == CblasNoTrans) {
    printf("NoTrans\n");
  } else {
    printf("Trans\n");
  }
  printf("m = %d\n", m);
  printf("n = %d\n", n);
  printf("alpha = %lf\n", alpha);
  printf("T_offset = %d\n", T_offset);
  printf("tensor_stride= %d\n", tensor_stride);
  printf("A_offset = %d\n", A_offset);
  printf("A_stride = %d\n", A_stride);
  printf("beta = %lf\n", beta);
  printf("output_col_stride = %d\n", output_col_stride);
  printf("output_stride = %d\n", output_stride);
  printf("-------------------------------\n");
}

/*
  MTTKRP_RowMajor();
  reorder_Factors();
  Description -- reorders the factor matrices for the
  MTTKRP_RowMajor function. Skips over the nth factor
  example skipping over 2 in
  {0,1,2,3,4}->{0,1,3,4}, also does the Dims.
*/
void reorder_Factors(ktensor *Y, double **reordered_Factors,
                     long int *reordered_Dims, long int n) {
  long int i, j;
  for (i = 0, j = 0; i < Y->nmodes; i++) {
    if (i != n) {
      reordered_Factors[j] = Y->factors[i];
      reordered_Dims[j] = Y->dims[i];
      j++;
    }
  }
}
/*
  MTTKRP_RowMajor();
  update_Partial_Hadamards();
*/
void update_Partial_Hadamards(ktensor *Y, long int *indexers,
                              double *partial_Hadamards,
                              double **reordered_Factors) {
  // check to see where we need to update form
  long int update_from, i;

  /*
    Iterate over the indexers array looking for a non 0.
    While the ith place is 0 keep searching, you will always hit a nonzero
    before the end because if the entries are all zero we have either just
    stared or just ended the KR product. update_from will hold the index of the
    last factor matrix to cycle, the one we should update from including that
    factor matrix.
  */
  for (i = 0; i < Y->nmodes - 1; i++) {
    update_from = i;
    if (indexers[i] != 0) break;
  }

  // if update_from is 0 we do not have do update any partials
  if (update_from == 0) {
    return;
  } else if (update_from >= Y->nmodes - 3) {
    /*
    If update_from is the second to last index or the last index,
    we have to update all of the partials
    */
    for (i = Y->nmodes - 4; i >= 0;
         i--) {  // start from the right most partial, Y->nmodes-4 is the last
                 // index of the partialHadamards array, i iterates over the
                 // partials array
      if (i == Y->nmodes - 4) {  // on the first iteration we need to access the
                                 // last and second factor matrix
        /*
          call vdMul to get the hadamard product of the last two factor
          matrices 1) length of the vectors 2) second to last factor matrix 3)
          last factor matrix 4) output location in partial_Hadmards
        */
        vdMul(Y->rank,
              &reordered_Factors[Y->nmodes - 3]
                                [Y->rank * indexers[Y->nmodes - 3]],
              &reordered_Factors[Y->nmodes - 2]
                                [Y->rank * indexers[Y->nmodes - 2]],
              &partial_Hadamards[Y->rank * i]);
      } else {  // we are not updating the last partial_Hadamard so we can use a
                // previous partial
        /*
          call vdMul to get the hadamards product of the current factor
          matrix based off i and the partial_Hadamards one stride to the right
          Note: the factor whose row we want is i-1
            1) the lenght of the vectors
            2) the (i+1)th factor matrices row
            3) the partial_Hadamard one stride to the right
            4) output location in patial_Hadamards
        */
        vdMul(Y->rank, &reordered_Factors[i + 1][Y->rank * indexers[i + 1]],
              &partial_Hadamards[Y->rank * (i + 1)],
              &partial_Hadamards[Y->rank * i]);
      }
    }
  } else {  // the case that we are updating from somewhere that is update_from
            // < Y->nmodes - 4, not the last or second to last
    for (i = update_from - 1; i >= 0; i--) {
      /*
        call vdMul to update the partial_Hadamards to the left of and
        including i
      */
      vdMul(Y->rank, &reordered_Factors[i + 1][Y->rank * indexers[i + 1]],
            &partial_Hadamards[Y->rank * (i + 1)],
            &partial_Hadamards[Y->rank * i]);
    }
  }
}

void KR_RowMajor(ktensor *Y, double *C, long int n) {
  long int i, j, lF, rF;
  if (n != Y->nmodes - 1)
    lF = Y->nmodes - 1;
  else
    lF = Y->nmodes - 2;
  rF = lF - 1;
  if (rF == n) rF--;

  for (i = 0; i < Y->dims[lF]; i++) {  // loop over the rows of the left matrix
    for (j = 0; j < Y->dims[rF];
         j++) {  // loop over the rows of the right matrix
      vdMul(Y->rank, &Y->factors[lF][i * Y->rank], &Y->factors[rF][j * Y->rank],
            &C[i * Y->rank * Y->dims[rF] + j * Y->rank]);
    }
  }
}

/*
  Multi_KR_RowMajor();
  1) Y, ktensor of factor matrices to for the KRP from
  2) C, output matrix
  3) n, modes to skip over only used fo the Y->nmodes == 3 case
*/

void Multi_KR_RowMajor(ktensor *Y, double *C, long int n) {
  if (Y->nmodes == 3) {
    KR_RowMajor(Y, C, n);
  } else {
    long int i, j, *indexers, *reordered_Dims;
    double *partial_Hadamards, **reordered_Factors;

    reordered_Factors =
        reinterpret_cast<double **>(malloc(sizeof(double *) * Y->nmodes - 1));
    reordered_Dims = (long int *)malloc(sizeof(long int) * Y->nmodes - 1);
    indexers = (long int *)malloc(sizeof(long int) * Y->nmodes -
                                  1);  // an indexer for each mode-1
    partial_Hadamards = reinterpret_cast<double *>(
        malloc(sizeof(double) * Y->rank * (Y->nmodes) -
               3));  // N-3 extra vectors, for partials

    // reorder the Factors and Dimensions
    reorder_Factors(Y, reordered_Factors, reordered_Dims, n);

    memset(indexers, 0, sizeof(long int) * (Y->nmodes - 1));

    // initialize the partial_Hadamards to the starting values, i indexes the
    // partial_Hadamards
    for (i = Y->nmodes - 4; i >= 0; i--) {
      if (i == Y->nmodes - 4) {
        vdMul(Y->rank, &reordered_Factors[Y->nmodes - 3][0],
              &reordered_Factors[Y->nmodes - 2][0],
              &partial_Hadamards[Y->rank * i]);
      } else {
        vdMul(Y->rank, &reordered_Factors[i + 1][0],
              &partial_Hadamards[Y->rank * (i + 1)],
              &partial_Hadamards[Y->rank * i]);
      }
    }

    // Iterate over the rows of the KRP matrix
    for (i = 0; i < Y->dims_product / Y->dims[n]; i++) {
      /*
      compute the current row using vdMul
        1) Y->rank, length of a row
        2) always some row of the 0th factor matrix
        3) always the first partial_hadamard
        4) the current row the KR matrix
      */
      vdMul(Y->rank, &reordered_Factors[0][Y->rank * indexers[0]],
            &partial_Hadamards[0], &C[Y->rank * i]);

      // update the indexers array
      for (j = 0; j < Y->nmodes - 1; j++) {
        indexers[j] += 1;
        if (indexers[j] >= reordered_Dims[j]) {
          indexers[j] = 0;
        } else {
          break;
        }
      }

      update_Partial_Hadamards(Y, indexers, partial_Hadamards,
                               reordered_Factors);
    }

    free(indexers);
    free(partial_Hadamards);
    free(reordered_Dims);
    free(reordered_Factors);
  }
}

// you need to include the diagonal
void Upper_Hadamard_RowMajor(long int nRows, long int nCols, double *A,
                             double *B, double *C) {
  long int i, j;

  for (i = 0; i < nRows; i++) {
    for (j = i; j < nCols; j++) {
      C[i * nCols + j] = A[i * nCols + j] * B[i * nCols + j];
    }
  }
}

/*
  CompareM()
  Description --
  A function for comparing two matrices to see if they differ
  significantly in some value. The last long intput option is your tollerance
  for the difference between A[i,j] and B[i,j]. if you make it a -1 the matlab
  eps*100 will be used. 1 -> true, 0 -> false
  Variables --
  A) matrix to compare against B
  B) matrix to compare against A nRows) the number of rows in A and B
  nCols) the number of colums in A and B
  eps) the tolerance for differences between A and B

*/
long int CompareM(double *A, double *B, long int nRows, long int nCols,
                  double eps) {
  if (eps == -1.0) eps = 2.220446049250313E-16;
  eps = eps * 100;
  double max_dif = eps;
  long int are_same = 1;  // 1 -> true, 0 -> false
  for (long int i = 0; i < nRows * nCols; i++) {
    if (fabs(A[i] - B[i]) > max_dif) {
      max_dif = A[i] - B[i];
      are_same = 0;
    }
  }
  printf("Max_dif=%.20lf\n", max_dif);
  return are_same;
}

void MTTKRP_RowMajor(tensor *T, double *K, double *C, long int rank,
                     long int n) {
  if (T == NULL) {
    printf("Tensor pointer is null, error\n");
    exit(-1);
  }
  if (K == NULL) {
    printf("Khatri Rhao product pointer is null, error\n");
    exit(-2);
  }
  if (C == NULL) {
    printf("Output Matrix is null!\n");
    exit(-3);
  }
  if (rank < 1) {
    printf("Rank must be greater than 1!\n");
    exit(-4);
  }
  if (n > T->nmodes - 1) {
    printf("n is larger than the number of modes in the tensor, T->nmodes!\n");
    exit(-5);
  }
  if (n < 0) {
    printf("n cannot be a negative number!\n");
    exit(-6);
  }

  // long int i, nDim;
  int i, nDim;
  double alpha, beta;

  // for calling dgemm_
  char nt = 'N';
  char t = 'T';
  int i_n = n;
  int i_rank = rank;

  if (n == 0) {
    // long int ncols = 1;
    int ncols = 1;

    ncols = T->dims_product / T->dims[n];
    alpha = 1.0;
    beta = 0.0;
    nDim = T->dims[n];
    /*
      This degemm call performs the column wise matrix multiply between
      the first mode matricization of the tensor and the matrix K. Operation is
      X_1 x K
      Inputs --
      1) CblasColMajor indicates the matrices are in column major format
      2) CblasNoTrans do not transpose the left matrix, T->data
      3) CblasNoTrans do not transpose the right matrix, K
      4) nDim, the number of rows in T-data
      5) rank, the number of columns in K
      6) ncols, the shared dimension, the number of columns in T->data
         and rows in K
      7) alpha
      8) T->data, the left matrix, first mode matricization of the tensor
      9) nDim, the leading dimension of T->data, the distance between
         columns of T->data
      10) K, the right matrix, khatri rao product
      11) ncols, the disctance between columns in the K matrix
      12) beta
      13) C, the output matrix,
      14) nDim the distance between columns in C
    */

    // printf("Input args to cblas_dgemm
    // function\nnDim=%d\nrank=%d\n,ncols=%d\n",*(T->dims+n-1), rank, ncols);
    // This does M*KR
    // cblas_dgemm( CblasRowMajor, CblasTrans, CblasNoTrans, nDim, rank, ncols,
    // alpha, T->data, nDim, K, rank, beta, C, rank );
    // This does KR'*M'
    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, rank, nDim, ncols,
    //             alpha, K, rank, T->data, nDim, beta, C, rank);
    dgemm_(&nt, &t, &i_rank, &nDim, &ncols, &alpha, K, &i_rank, T->data, &nDim,
           &beta, C, &i_rank);

  } else {  // if n != 0 it is not the first dimension, so n is at least 1
    int nmats = 1;  // nmats is the number of submatrices to be multiplied
    int ncols = 1;  // ncols is the number of columns in a submatrix

    // calculate the number of columns in the sub-matrix of a matricized tensor
    for (i = 0; i < n; i++) {
      ncols *= T->dims[i];
    }

    // calculate the number of row major sub-matrices in the matricized tensor
    for (i = n + 1; i < T->nmodes; i++) {
      nmats *= T->dims[i];
    }

    // do nmats dgemm calls on chunks of our two matrices
    for (i = 0; i < nmats; i++) {
      if (i == 0)
        beta = 0.0;
      else
        beta = 1.0;
      alpha = 1.0;
      nDim = T->dims[n];

      // printf("Input args to cblas_dgemm
      // function\nnDim=%d\nrank=%d\ncols=%d\nnmats=%d\ni=%d\n",nDim, rank,
      // ncols, nmats, i);

      /*
        This dgemm call is more complex
        1) CblasColMajor - treat matrices as if they are in column major
           order
        2) CblasTrans - transpose the left matrix because we want it in
           row major ordering
        3) CblasNoTrans - still treat K as a column major
           matrix
        4) nDim - the number of rows in a submatrix
        5) rank - the number of columns in K
        6) ncols - the number of columns in a submatrix and rows in K
        7) alpha
        8) T->data + i*ncols*nDim, ncols*nDim is the size of
           a submatrix, a submatix is stored in contiguous memory,
           T->datancols*nDims*i indicates which submatrix we are on,
        9) ncols -
           the distance between rows of a submatrix, but remember its transposed
           so it could also be thought of as the number distance between columns
           of a transposed submatrix.
        10) K+i*ncols - starting polong int of the khatri rao submatrix
        11) ncols*nmats - the distance between columns of
         the K  matrix, ncols*nmats is the number of rows in the full K matrix
        12) beta
        C) the out put matrix, size nDim by rank
        nDim) the distance between columns of the C matrix
      */
      // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nDim, rank,
      // ncols,
      //             alpha, T->data + i * nDim * ncols, ncols,
      //             K + i * ncols * rank, rank, beta, C, rank);
      dgemm_(&nt, &nt, &nDim, &i_rank, &ncols, &alpha,
             T->data + i * nDim * ncols, &ncols, K + i * ncols * rank, &i_rank,
             &beta, C, &i_rank);
    }
  }  // End of else
}

/*
  MHada_RowMajor
  This function is a simpler version of MHada_RowMajor.
  It computes the Hadamards product of n-1 matrices skipping over the nth
  matrix in the array of matrices that it is given.
*/
void MHada_RowMajor(ktensor *Y, double **SYRKs, double *V, long int n) {
  long int i, j, c;
  j = 0;

  if (n ==
      0)  // if n==0 we do not want to use the 0th factor matrix so start at 1
    i = 1;
  else
    i = 0;

  for (c = 0; c < Y->nmodes - 1;
       c++, i++) {  // c keeps track of the number of multiplies that should be
                    // performed
    if (i == n)     // skip over the nth factor matrix
      i++;
    /*
      For the first round of multiples we just need to copy C long into
      V, this avoids having to fill V with any initial values.
    */

    if (j == 0) {  // couldn't I just use c?
      j += 1;
      cblas_dcopy(Y->rank * Y->rank, SYRKs[i], 1, V, 1);
    } else {
      Upper_Hadamard_RowMajor(Y->rank, Y->rank, V, SYRKs[i], V);
    }
  }  // end of for i
}

void do_SYRKs_RowMajor(ktensor *Y, double **SYRKs) {
  long int i;
  double alpha, beta;
  alpha = 1.0;
  beta = 0.0;
  for (i = 0; i < Y->nmodes; i++) {
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, Y->rank, Y->dims[i],
                alpha, Y->factors[i], Y->rank, beta, SYRKs[i], Y->rank);
  }
}

/*
  normalize_Factor_Matrix_RowMajor();
  Description --
  Takes a ktensor and an long int n and normalizes the nth factor
  matrix. This will over write the lambda values of the ktensor with the
  column norms of the nth factor matrix. Variables -- Y) the ktensor n) the
  factor matrix to normalize i) general purpose iterator, moves over the rank
  of the nth factor matrix of Y

  Notes: ask Grey about the 1.0/Y-labmdas[i] division operation, worrying
  about divide by 0 errors!
*/
void normalize_Factor_Matrix_RowMajor(ktensor *Y, long int n) {
  long int i;
  for (i = 0; i < Y->rank; i++) {
    Y->lambdas[i] = cblas_dnrm2(Y->dims[n], Y->factors[n] + i, Y->rank);
    cblas_dscal(Y->dims[n], 1.0 / Y->lambdas[i], Y->factors[n] + i, Y->rank);
  }
}

/*
  normalize_Ktensor_RowMajor()
  Takes in a ktensor Y whose factor matrices are stored in row major
  ordering and normalizes them column-wise saving the weights in the Y->lambdas
  vector.
*/
void normalize_Ktensor_RowMajor(ktensor *Y) {
  long int i, j;
  double l;

  for (i = 0; i < Y->nmodes; i++) {  // loop over the factor matrices
    for (j = 0; j < Y->rank;
         j++) {  // loop over the columns of the factor matrices
      if (i == 0) {
        Y->lambdas[j] = cblas_dnrm2(Y->dims[i], Y->factors[i] + j, Y->rank);
        if (Y->lambdas[j] == 0.0) {
          exit(-8);
        }
        cblas_dscal(Y->dims[i], 1.0 / Y->lambdas[j], Y->factors[i] + j,
                    Y->rank);
      } else {
        l = cblas_dnrm2(Y->dims[i], Y->factors[i] + j, Y->rank);
        if (l == 0.0) {
          exit(-8);
        }
        Y->lambdas[j] *= l;
        cblas_dscal(Y->dims[i], 1.0 / l, Y->factors[i] + j, Y->rank);
      }
    }
  }
}

/*
TransposeM()
  Description --
    Transposes an input matrix A and stores it in A_T
  Variables --
    A) matrix to transpose
    A_T) output matrix
    n) the number of rows in A and the number of columns in A_T
    m) the number of columms in A and the number of rows in A_T
    i) iterator
    j) iterator

Notes: Perhaps just store it in the old matrix?
*/
void TransposeM(double *A, double *A_T, long int rowsA, long int colsA) {
  long int i;
  for (i = 0; i < rowsA; i++) {
    /*
      m) copy m items each time
      A
      n) the distance between elements in a row of A
      A_T)
      1) the distance between elements in a row of A_T
    */
    cblas_dcopy(colsA, A + i, rowsA, A_T + colsA * i, 1);
  }
}

/*
Full_nMode_Matricization_RowMajor();
  Description -- creates the full nmode matricization of a tensor
  from the factor matrices of a Ktensor. This assumes the intputs are in
  RowMajor order and returns the matricization in ColumnMajor ordering
  F_n(KR!=n)^T
*/
void Full_nMode_Matricization_RowMajor(tensor *T, ktensor *Y, long int n) {
  double *KR, alpha, beta;

  KR = reinterpret_cast<double *>(
      malloc(sizeof(double) * (Y->dims_product / Y->dims[n]) * Y->rank));

  Multi_KR_RowMajor(Y, KR, n);

  alpha = 1.0;
  beta = 0.0;
  // cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Y->dims[n],
  //             (Y->dims_product / Y->dims[n]), Y->rank, alpha, Y->factors[n],
  //             Y->rank, KR, Y->rank, beta, T->data, Y->dims[n]);
  int i_m = Y->dims[n];
  int i_n = (Y->dims_product / Y->dims[n]);
  int i_k = Y->rank;
  char t = 'T';
  char nt = 'N';
  dgemm_(&t, &nt, &i_m, &i_n, &i_k, &alpha, Y->factors[n], &i_k, KR, &i_k,
         &beta, T->data, &i_m);

  free(KR);
}

/*
approximation_Error()
  Computes the norm of a tensor and its current approximation given a KRP
  and a factor matrix
*/
double approximation_Error(double *X, double *KR, ktensor *Y, long int n) {
  // I think you only wanna do this for n == 0 but idk, you could easily do it
  // for the last one also
  double alpha, beta;
  alpha = 1.0;
  beta = -1.0;

  // now we have X = X-Y, where Y is the approximation of X
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Y->dims[n],
              (Y->dims_product / Y->dims[n]), Y->rank, alpha, Y->factors[n],
              Y->rank, KR, Y->rank, beta, X, Y->dims[n]);

  return cblas_dnrm2(Y->dims_product, X, 1);
}

/**
  CP_ALS_efficient_error_computation();
  needs to compute dotXX - 2*dot XY + dotYY
  1) Y, ktensor
  2) n, mode that was just updated or is being updated
  3) MTTKRP, the MTTKRP of the nth mode
  4) V, the multiple hadamard product for the nth mode
  5) S, pointer to the storage space for the SYRK of the nth factor matrix
  6) tensor_norm, the fro norm of the tensor input.
*/
double CP_ALS_efficient_error_computation(ktensor *Y, long int n,
                                          double *MTTKRP, double *V, double *S,
                                          double tensor_norm) {
  double dotXY, dotYY, *lambda_array, e;
  lambda_array = reinterpret_cast<double *>(malloc(sizeof(double) * Y->rank));

  dotXY = cblas_ddot(Y->dims[n] * Y->rank, Y->factors[n], 1, MTTKRP, 1);

  normalize_Factor_Matrix_RowMajor(Y, n);

  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, Y->rank, Y->dims[n], 1.0,
              Y->factors[n], Y->rank, 0.0, S, Y->rank);

  Upper_Hadamard_RowMajor(Y->rank, Y->rank, V, S, V);

  cblas_dsymv(CblasRowMajor, CblasUpper, Y->rank, 1.0, V, Y->rank, Y->lambdas,
              1, 0.0, lambda_array, 1);

  dotYY = cblas_ddot(Y->rank, lambda_array, 1, Y->lambdas, 1);

  e = ((tensor_norm * tensor_norm) - (2 * dotXY) + dotYY);

  free(lambda_array);

  return e;
}

/**
Computes the error between T and Y. ||T->data - Y||_2
  Does so by forming the explicit approximatino of T form Y
  subtracting the two matricized tensors and taking the 2norm.
*/
double CP_ALS_naive_error_computation(tensor *T, ktensor *Y, double *X,
                                      double *KRP, long int n) {
  double e;

  cblas_dcopy(T->dims_product, T->data, 1, X, 1);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Y->dims[n],
              (Y->dims_product / Y->dims[n]), Y->rank, 1.0, Y->factors[n],
              Y->rank, KRP, Y->rank, -1.0, X, Y->dims_product / Y->dims[n]);
  e = cblas_dnrm2(Y->dims_product, X, 1);

  return e;
}

void dims_check(long int *dims, long int length) {
  /*
  This function takes an integer array of dimension and the array's
  length. All entires in dims should be greater than 0 and length should be 1
  or greater.
  */

  long int i;

  if (length < 1) {
    printf("From dims_check():: length < 1, Exit(-1)");
    exit(-1);
  }
  if (dims == NULL) {
    printf("From dims_check():: dims == NULL, Exit(-2)");
    exit(-2);
  }
  for (i = 0; i < length; i++) {
    if (dims[i] < 1) {
      printf("From dims_check():: dims[%d] < 1, Exit(-3)", i);
      exit(-3);
    }
  }
}

/*
reoder_Ktensor();
  Reoders the factor matrices and dimension of a ktensor by removing the
  nth mode. Only shuffles polong inters, no deep copies of factor matrices are
  made.
  1) Y the original ktnensor
  2) nY the kentsor without the nth mode
  Example: [0,1,2,3,4], n = 3
           [0,1,2,4]
*/
void reorder_Ktensor(ktensor *Y, ktensor *nY, long int n) {
  long int i, j;
  nY->nmodes = Y->nmodes - 1;
  nY->factors =
      reinterpret_cast<double **>(malloc(sizeof(double *) * nY->nmodes));
  nY->dims = (long int *)malloc(sizeof(long int) * nY->nmodes);
  nY->lambdas = reinterpret_cast<double *>(malloc(sizeof(double) * Y->rank));
  nY->rank = Y->rank;
  nY->dims_product = Y->dims_product / Y->dims[n];

  for (i = 0, j = 0; i < Y->nmodes; i++) {
    if (i != n) {
      nY->factors[j] = Y->factors[i];
      nY->dims[j] = Y->dims[i];
      j++;
    }
  }
}
/*
void compute_KRP_Indices();
  Computes the factor matrix indeces for a given row in a khatri rao
  product. The indeces are in reverse order so as to be in keeping with the
  reverse order khatri rao product functions used here.
*/
void compute_KRP_Indices(long int j, ktensor *Y, long int *indeces) {
  long int i, p;
  p = Y->dims_product;

  for (i = Y->nmodes - 1; i >= 0; i--) {
    p = p / Y->dims[i];

    if (p != 0) indeces[i] = j / p;
    j -= indeces[i] * p;
  }
  // printf("nmodes = %d\n",Y->nmodes);
  // printf("Indeces = ");
  // for( i = 0; i < Y->nmodes; i++){
  //    printf("%d,",indeces[i]);
  //}
  // printf("\n");
  // Y->nmodes = Y->nmodes + 1;
}

/**
Wrapper function for the reverse KRP of a ktensor Y
  1) Ktensor object
  2) number of threads to use
  3) KRP, output KRP
*/
void wrapper_Parallel_Multi_revKRP(ktensor *Y, long int num_threads,
                                   double *KRP) {
  long int useful_num_threads =
      num_threads;  // The number of threads that shuould be used <= num_threads

  if (Y->dims_product < num_threads) useful_num_threads = Y->dims_product;

#pragma omp parallel num_threads(useful_num_threads)
  {
    long int thread_id = omp_get_thread_num();
    long int start, end;

    compute_Iteration_Space(thread_id, useful_num_threads, Y->dims_product,
                            &start, &end);
    parallel_Multi_KR_RowMajor(Y, &KRP[start * Y->rank], start, end);
  }
}

/*
Gen_Tensor();
  This function generates a Ktensor using uniformily random numbers. Given
  a Rank, a number of dimensions, and the sizes of those dimensiosn random
  factor matrices are generated and then a full tensor is created from those.
  All memory for the objects is allocated inside of the function.
  Call clean_Up_Gen_Tensor() to free all the memory when finished.
*/

void Gen_Tensor(ktensor *Y, tensor *T, long int R, long int N, long int *D,
                double noise) {
  /*
    A function for returning a randomly generated Rank R and N mode tensor
    this function will allocate all of the memory needed for the tensor
    just passing the declared ktensor and tensor objects.
    1) Y, a ktensor
    2) T, a tensor
    3) N, the number of modes in the
    4) R, the rank of the tensor
    5) D, the length of the dimensions of the tensor, D is length N
  */

  long int i, j;
  double alpha, tensor_norm;

  srand(time(NULL));

  // Set up the ktensor Y
  Y->nmodes = N;
  Y->rank = R;
  Y->dims = (long int *)malloc(sizeof(long int) * Y->nmodes);
  Y->factors =
      reinterpret_cast<double **>(malloc(sizeof(double *) * Y->nmodes));
  for (i = 0; i < Y->nmodes; i++) Y->dims[i] = D[i];

  Y->dims_product = 1;
  for (i = 0; i < Y->nmodes; i++) {
    Y->factors[i] = reinterpret_cast<double *>(
        malloc(sizeof(double) * Y->rank * Y->dims[i]));
    Y->dims_product *= Y->dims[i];
    for (j = 0; j < Y->rank * Y->dims[i]; j++) {
      Y->factors[i][j] =
          (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 2 - 1;
    }
  }
  Y->lambdas = reinterpret_cast<double *>(malloc(sizeof(double) * Y->rank));
  memset(Y->lambdas, 0, sizeof(double) * Y->rank);

  // Set up the tensor T
  T->nmodes = N;
  T->dims = (long int *)malloc(sizeof(long int) * Y->nmodes);
  T->data =
      reinterpret_cast<double *>(malloc(sizeof(double) * Y->dims_product));
  T->dims_product = Y->dims_product;

  for (i = 0; i < T->nmodes; i++) T->dims[i] = D[i];

  Full_nMode_Matricization_RowMajor(T, Y, 0);

  tensor_norm = cblas_dnrm2(Y->dims_product, T->data, 1);

  if (noise > 0.0) {
    alpha = (noise * tensor_norm) / sqrt(Y->dims_product);

    // add alpha * (a normally random number to T->data)
    double U1, U2, Z1, Z2;
    for (i = 0; i < Y->dims_product - 1; i += 2) {
      // Generate 2 uniformly random numbers in range 0 to 1
      U1 = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
      U2 = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

      // Transform to 2 normally distributed numbers in range 0 to 1
      Z1 = sqrt(-2.0 * log(U1)) * cos(2 * M_PI * U2);
      Z2 = sqrt(-2.0 * log(U1)) * sin(2 * M_PI * U2);

      // Move then to the range -1 to 1
      Z1 = (Z1 * 2.0) - 1.0;
      Z2 = (Z2 * 2.0) - 1.0;

      T->data[i] += alpha * Z1;
      T->data[i + 1] += alpha * Z2;
    }
    if ((Y->dims_product % 2) == 0) {
      // do the last element
      // Generate 2 uniformly random numbers in range 0 to 1
      U1 = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
      U2 = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

      // Transform to 2 normally distributed numbers in range 0 to 1
      Z1 = sqrt(-2.0 * log(U1)) * cos(2 * M_PI * U2);

      // Move then to the range -1 to 1
      Z1 = (Z1 * 2.0) - 1.0;

      T->data[Y->dims_product - 1] += alpha * Z1;
    }
  }
}

/*
parallel_Multi_KR_RowMajor()
Computes a RowMajor KPR limited by start and end which are row indeces
*/
void parallel_Multi_KR_RowMajor(ktensor *Y, double *C, long int start,
                                long int end) {
  if (Y->nmodes == 2) {
    /*
      If only 2 factor matrices, then no partials are used call this
      Function instead
    */
    parallel_KR_RowMajor(Y, C, start, end);
  } else {
    long int i, j, *indexers, nmodes;
    double *partial_Hadamards;

    nmodes = Y->nmodes;

    indexers = (long int *)malloc(sizeof(long int) *
                                  nmodes);  // An indexer for each mode-1
    partial_Hadamards = reinterpret_cast<double *>(
        malloc(sizeof(double) * Y->rank *
               (nmodes)-2));  // N-3 extra vectors, for storing
                              // inermediate hadamard prducts

    memset(indexers, 0, sizeof(long int) * (nmodes));

    // Compute the starting indices
    compute_KRP_Indices(start, Y, indexers);

    // Initialize the partial_Hadamards to the starting values, i indexes the
    // partial_Hadamards Hadamard the last two factor matrices and stores them
    // in the last partial hadamard
    vdMul(Y->rank, &Y->factors[nmodes - 2][Y->rank * indexers[nmodes - 2]],
          &Y->factors[nmodes - 1][Y->rank * indexers[nmodes - 1]],
          &partial_Hadamards[Y->rank * (nmodes - 3)]);

    // For each other partial Hadamard multiply the previous partial Hadamard
    // with a row of a factor matrix
    for (i = nmodes - 4; i >= 0; i--)
      vdMul(Y->rank, &Y->factors[i + 1][Y->rank * indexers[i + 1]],
            &partial_Hadamards[Y->rank * (i + 1)],
            &partial_Hadamards[Y->rank * i]);

    // Iterate over the rows of the KRP matrix
    for (i = start; i < end; i++) {
      /*
        compute the current row using vd mul, multiply first partial
        hadamard with row of first factor matrix 1) length of a row 2) always
        some row of the 0th factor matrix 3) always the first partial_hadamard
        4) the current row the KR matrix

      */
      vdMul(Y->rank, &Y->factors[0][Y->rank * indexers[0]],
            &partial_Hadamards[0], &C[Y->rank * (i - start)]);

      // update the indexers array
      for (j = 0; j < nmodes; j++) {
        indexers[j] += 1;
        if (indexers[j] >= Y->dims[j]) {
          indexers[j] = 0;
        } else {
          break;
        }
      }

      parallel_update_Partial_Hadamards(Y, indexers, partial_Hadamards);
    }
    free(indexers);
    free(partial_Hadamards);
  }
}

/**
  parallel_KR_RowMajor();
  Does the parallel KRP of the factor matrices in the ktensor Y
*/
void parallel_KR_RowMajor(ktensor *Y, double *C, long int start, long int end) {
  long int i, j, *indexers, c;

  indexers = (long int *)malloc(sizeof(long int) * Y->nmodes);
  memset(indexers, 0, sizeof(long int) * (Y->nmodes));

  compute_KRP_Indices(start, Y, indexers);

  c = start;
  i = indexers[1];
  j = indexers[0];
  for (; (c < end) && (i < Y->dims[1]); i++) {
    if (c != start) j = 0;
    for (; (c < end) && (j < Y->dims[0]); j++) {
      /*
        vdMul call
        1) number of elements to multiply together
        2) pointer to first vector
        3) pointer to second vector
        4) pointer to output location
           -i*Y->rank*Y->dims[0], the size of the right matrix times
           i -j*Y->rank, the length of a row times j
      */
      vdMul(Y->rank, &Y->factors[1][i * Y->rank], &Y->factors[0][j * Y->rank],
            &C[(c - start) * Y->rank]);
      c++;
    }
  }
  free(indexers);
}

void parallel_update_Partial_Hadamards(ktensor *Y, long int *indexers,
                                       double *partial_Hadamards) {
  long int update_from, i;
  long int nmodes = Y->nmodes;

  for (i = 0; i < nmodes; i++) {
    update_from = i;
    if (indexers[i] != 0) break;
  }

  // if update_from is 0 we do not have do update any partials
  if (update_from == 0) {
    return;
  } else if (update_from >= nmodes - 2) {
    /*
    If update_from is the second to last index or the last index,
    we have to update all of the partials, including combining the last
    two factor matrices seperately
    */
    /*
    call vdMul to get the hadamard product of the last two factor
    matrices 1) length of the vectors 2) second to last factor matrix 3) last
    factor matrix 4) output location in partial_Hadmards
    */
    vdMul(Y->rank, &Y->factors[nmodes - 2][Y->rank * indexers[nmodes - 2]],
          &Y->factors[nmodes - 1][Y->rank * indexers[nmodes - 1]],
          &partial_Hadamards[Y->rank * (nmodes - 3)]);
    // Update the rest of the partial Hadamards
    for (i = nmodes - 4; i >= 0; i--) {  // i iterates over the partials array
      /*
        call vdMul to get the hadamards product of the current factor
        matrix based off i and the partial_Hadamards one stride to the right
        Note: the factor whose row we want is i-1
          1) the lenght of the vectors
          2) the (i+1)th factor matrices row
          3) the partial_Hadamard one stride to the right
          4) output location in patial_Hadamards
      */
      vdMul(Y->rank, &Y->factors[i + 1][Y->rank * indexers[i + 1]],
            &partial_Hadamards[Y->rank * (i + 1)],
            &partial_Hadamards[Y->rank * i]);
    }
  } else {  // the case that we are updating from somewhere that is update_from
            // < Y->nmodes - 4, not the last or second to last
    for (i = update_from - 1; i >= 0; i--) {
      /*
      call vdMul to update the partial_Hadamards to the left of and
      including i
      */
      vdMul(Y->rank, &Y->factors[i + 1][Y->rank * indexers[i + 1]],
            &partial_Hadamards[Y->rank * (i + 1)],
            &partial_Hadamards[Y->rank * i]);
    }
  }
}

/*
 compute_Iteration_Space()
  Computes the iteration space a given thread should work on.
  1) thread_id
  2) total number of threads
  3) stat of the thread's space
  4) end of the thread's space
  5) space to distribute
*/
void compute_Iteration_Space(long int thread_id, long int num_threads,
                             long int iter_space, long int *start,
                             long int *end) {
  /*
          1) b,     block size
          2) limit, bound on thread_ids who get +1 on block size
  */
  long int b, limit;

  limit = iter_space % num_threads;
  b = iter_space / num_threads;

  if (thread_id < limit) {
    *start = thread_id * (b + 1);
    *end = *start + (b + 1);
  } else {
    *start = (limit * (b + 1)) + ((thread_id - limit) * b);
    *end = *start + b;
  }
}

/*
  Function for freeing the  memory of ktnensor and tensor objects
*/
void clean_Up_Gen_Tensor(ktensor *Y, tensor *T) {
  long int i;

  // Ktensor Y memory
  for (i = 0; i < Y->nmodes; i++) {
    free(Y->factors[i]);
  }
  free(Y->factors);
  free(Y->dims);
  free(Y->lambdas);

  // Tensor T memory
  free(T->data);
  free(T->dims);
}

/*
  1) rank
  2) noise
  3) num_threads
  4) nmodes
  5) array of dimensions
*/
void process_inputs(long int argc, char *argv[], tensor_inputs *inputs) {
  long int i, j = 1;
  long int offset;  // 0)program_name 1)rank, 2)num_threads, 3)nmodes, 4)
                    // max_iters, 5)eps

  inputs->rank = atoi(argv[j++]);  // 1
  // inputs->noise = atof(argv[2]);
  inputs->num_threads = atoi(argv[j++]);  // 2
  inputs->max_iters = atoi(argv[j++]);    // 3
  inputs->tolerance = atof(argv[j++]);    // 4
  inputs->nmodes = atoi(argv[j++]);       // 5

  offset = j;

  printf("N=%d\n", inputs->nmodes);
  inputs->dims = (long int *)malloc(sizeof(long int) * inputs->nmodes);
  for (i = 0; i < inputs->nmodes; i++) inputs->dims[i] = atoi(argv[offset + i]);

  if (inputs->num_threads < 1) {
    printf(
        "From process_inputs(), num_threads = %d, num_threads must be at least "
        "1\nExit\n",
        inputs->num_threads);
    exit(-1);
  }
  if (inputs->rank < 1) {
    printf("From process_inputs(), rank = %d, rank must be at least 1\nExit\n",
           inputs->rank);
    exit(-1);
  }
  dims_check(inputs->dims, inputs->nmodes);

  printf("****************************************\n");
  printf("Processed inputs:\n");
  printf("inputs->rank = %d\n", inputs->rank);
  printf("inputs->num_threads = %d\n", inputs->num_threads);
  printf("inputs->max_iters = %d\n", inputs->max_iters);
  printf("inputs->tolerance = %.20lf\n", inputs->tolerance);
  printf("inputs->nmodes = %d\n", inputs->nmodes);
  printf("inputs->dims[%d", inputs->dims[0]);
  for (i = 1; i < inputs->nmodes; i++) {
    printf(",%d", inputs->dims[i]);
  }
  printf("]\n");
  printf("****************************************\n");
}

void destroy_inputs(tensor_inputs *inputs) { free(inputs->dims); }

/*
  LR_Ktensor_Reodering_newY()
  This function is for reordering the factor matrices of a given Ktensor
  It takes the left or right factor matrices in relation to some index
  Mallocs memory that needs to be free'd later on.
*/
void LR_Ktensor_Reordering_newY(ktensor *Y, ktensor *nY, long int n,
                                direction D) {
  long int i, jump;

  if (D == ::direction::left) {
    nY->nmodes = n;
    jump = 0;
  } else {
    nY->nmodes = Y->nmodes - (n + 1);
    jump = n + 1;
  }

  nY->factors =
      reinterpret_cast<double **>(malloc(sizeof(double *) * nY->nmodes));
  nY->dims = (long int *)malloc(sizeof(long int) * nY->nmodes);
  nY->rank = Y->rank;
  nY->lambdas = reinterpret_cast<double *>(malloc(sizeof(double) * nY->rank));

  for (i = 0; i < nY->nmodes; i++) {
    nY->factors[i] = Y->factors[i + jump];
    nY->dims[i] = Y->dims[i + jump];
  }
  for (i = 0; i < nY->rank; i++) nY->lambdas[i] = Y->lambdas[i];
  for (i = 0, nY->dims_product = 1; i < nY->nmodes; i++)
    nY->dims_product *= nY->dims[i];
}

/*
  LR_Ktensor_Reodering_newY()
  This function is for reordering the factor matrices of a given Ktensor.
  It takes the left or right factor matrices in relation to some index.
  Does not allocate any new memory.
*/
void LR_Ktensor_Reordering_existingY(ktensor *Y, ktensor *nY, long int n,
                                     direction D) {
  long int i, jump;

  if (D == ::direction::left) {
    nY->nmodes = n;
    jump = 0;
  } else {
    nY->nmodes = Y->nmodes - (n + 1);
    jump = n + 1;
  }

  nY->rank = Y->rank;
  nY->dims_product = 1;
  for (i = 0; i < nY->nmodes; i++) {
    nY->factors[i] = Y->factors[i + jump];
    nY->dims[i] = Y->dims[i + jump];
    nY->dims_product *= nY->dims[i];
  }
  for (i = 0; i < nY->rank; i++) nY->lambdas[i] = Y->lambdas[i];
}

/*
  Removes a mode from an existing ktensor.
  Does NOT reset the memory.
*/
void remove_mode_Ktensor(ktensor *Y, long int n) {
  long int i, j;

  if (n < 0 || n >= Y->nmodes) {
    printf("In remove_mode_Ktensor() invalid value of n == %d\nExit\n", n);
    exit(-1);
  }

  Y->nmodes -= 1;
  for (i = 0, j = 0; i < Y->nmodes + 1; j++) {
    if (j == n) {
      // do nothing
    } else {
      Y->dims[i] = Y->dims[j];
      Y->factors[i] = Y->factors[j];
      i++;
    }
  }
}

/*
  Destroys a given ktensor object
  clear_factors, is a flag that whne == 1 will tell the funciton to free
  the factor matrices also.
*/
void destruct_Ktensor(ktensor *Y, long int clear_factors) {
  long int i;

  if ((clear_factors != 0) && (clear_factors != 1)) {
    printf(
        "clear_factors argument in destruct_Ktensor() was not 0 or 1. Return. "
        "%d\n",
        clear_factors);
    return;
  }

  if (clear_factors == 1) {
    for (i = 0; i < Y->nmodes; i++) free(Y->factors[i]);
  } else {
    for (i = 0; i < Y->nmodes; i++) Y->factors[i] = NULL;
  }

  free(Y->factors);
  free(Y->dims);
  free(Y->lambdas);
}

void destruct_Tensor(tensor *T) {
  free(T->data);
  free(T->dims);
}

/*
  swaps the data for tensor t1 and t2
*/
void tensor_data_swap(tensor *t1, tensor *t2) {
  double *M;
  M = t1->data;
  t1->data = t2->data;
  t2->data = M;
}

/*

*/
void LR_tensor_Reduction(tensor *T, tensor *nT, long int n, direction D) {
  long int i, jump;

  if (D == ::direction::left) {
    jump = 0;
    nT->nmodes = n;
  } else {  // D == ::direction::right
    jump = n + 1;
    nT->nmodes = T->nmodes - jump;
  }

  // nT->data, this function doest not deal with the data attribute
  // use void tensor_data_swap to affect data

  nT->dims_product = 1;
  for (i = 0; i < nT->nmodes; i++) {
    nT->dims[i] = T->dims[i + jump];
    nT->dims_product *= nT->dims[i];
  }
}

/*
  Creates a copy of a ktensor object and stores it in
  the given address.
*/
void ktensor_copy_constructor(ktensor *Y, ktensor *nY) {
  long int i;

  nY->nmodes = Y->nmodes;
  nY->rank = Y->rank;

  nY->dims = (long int *)malloc(sizeof(long int) * nY->nmodes);
  nY->factors =
      reinterpret_cast<double **>(malloc(sizeof(double *) * nY->nmodes));
  nY->lambdas = reinterpret_cast<double *>(malloc(sizeof(double) * nY->rank));

  nY->dims_product = 1;
  for (i = 0; i < nY->nmodes; i++) {
    nY->dims[i] = Y->dims[i];
    nY->factors[i] = Y->factors[i];
    nY->dims_product *= nY->dims[i];
  }

  cblas_dcopy(nY->rank, Y->lambdas, 1, nY->lambdas, 1);
}

direction opposite_direction(direction D) {
  if (D != ::direction::left && D != ::direction::right) {
    printf("In opposite_direction input D was invalid!\nExit\n");
    exit(-1);
  }

  if (D == ::direction::left) {
    return ::direction::right;
  } else {
    return ::direction::left;
  }
}

#endif  // DIMTREE_DDTTENSOR_HPP_
