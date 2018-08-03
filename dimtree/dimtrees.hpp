/* Copyright Koby Hayashi 2018 */

#ifndef DIMTREE_DIMTREES_HPP_
#define DIMTREE_DIMTREES_HPP_

#include "dimtree/ddttensor.hpp"
#include "dimtree/dimtrees.h"

/**
        Performs a tensor times a matrix based on some split of dimensions.
        - Assumptions
                - dimensions splits must be contiguous
                - you can only multiply by a row major matrix
                - tensors are column major in the first matricization
        - Cases
                - multiply on the left
                - multiply on the right
                - output matrix is row major
                - output matrix is col major
        - Inputs
                - OL, Output_Layout, specifies if the output should be row or
   col major
                - s, a split dimension index, s is included in the left side
                - D, specifies which side of s will be the leading dimensions of
   the matricized tensor
                - r, specifies the number of columns in A
                - T, a tensor to multiply by
                - A, a matrix to multiply against the tensor
                - C, an output matrix
*/
void partial_MTTKRP(Output_Layout OL, long int s, direction D, tensor *T,
                    double *A, long int r, double *C, long int num_threads) {
  // mkl_set_num_threads(num_threads);
  // openblas_set_num_threads(num_threads);

  CBLAS_ORDER dgemm_layout;  // layout for dgemm calls
  // CBLAS_TRANSPOSE trans_tensor, trans_A;
  // dgemm related variables.
  char trans_tensor, trans_A;

  int i, m, n, k, output_stride, tensor_stride, right_dims_product,
      left_dims_product, i_r;
  double alpha, beta;

  alpha = 1.0;
  beta = 0.0;

  if (s < 0 || s >= T->nmodes - 1) {  // s cannot be negative or be equal to N-1
                                      // or be greater than N-1
    printf("Invalid value of s in partial_MTTKRP(), s = %d\nExit\n", s);
    exit(-1);
  }
  if (T == NULL) {
    printf("T == NULL in partial_MTTKRP()\nExit\n");
    exit(-1);
  }
  if (A == NULL) {
    printf("A == NULL in partial_MTTKRP()\nExit\n");
    exit(-1);
  }
  if (C == NULL) {
    printf("C == NULL in partial_MTTKRP()\nExit\n");
    exit(-1);
  }

  right_dims_product = 1;
  left_dims_product = 1;  // the left dimension product always includes s

  for (i = 0; i < s + 1; i++) left_dims_product *= T->dims[i];
  for (i = s + 1; i < T->nmodes; i++) right_dims_product *= T->dims[i];

  if (D == ::direction::left) {
    // m = right_dims_product;  // Tensor->data is m x k, A is k x n
    //                          // n = r;
    // k = left_dims_product;
    // tensor_stride = left_dims_product;
    if (OL == RowMajor) {
      dgemm_layout = CblasRowMajor;
      // trans_tensor = CblasNoTrans;
      // trans_A = CblasNoTrans;
      trans_tensor = 'N';
      trans_A = 'N';
      m = r;
      n = right_dims_product;
      k = left_dims_product;
      tensor_stride = left_dims_product;
      output_stride = r;
      dgemm_(&trans_A, &trans_tensor, &m, &n, &k, &alpha, A, &m, T->data, &tensor_stride,
             &beta, C, &output_stride);
      // assuming dgemm. comment if not neccessary
    } else {
      // trans_tensor = CblasTrans;
      // trans_A = CblasTrans;
      dgemm_layout = CblasColMajor;
      trans_tensor = 'T';
      trans_A = 'T';
      output_stride = right_dims_product;
      m = right_dims_product;  // Tensor->data is m x k, A is k x n
                               // n = r;
      k = left_dims_product;
      i_r = r;
      tensor_stride = left_dims_product;
      dgemm_(&trans_tensor, &trans_A, &m, &i_r, &k, &alpha, T->data,
             &tensor_stride, A, &i_r, &beta, C, &output_stride);
    }
  } else {  // D == right
    // m = left_dims_product;
    // // n = r;
    // k = right_dims_product;
    tensor_stride = left_dims_product;
    if (OL == RowMajor) {
      dgemm_layout = CblasRowMajor;
      // trans_tensor = CblasTrans;
      trans_tensor = 'T';
      // trans_A = CblasNoTrans;
      trans_A = 'N';
      m = r;
      n = left_dims_product;
      k = right_dims_product;
      output_stride = r;
      tensor_stride = left_dims_product;
      dgemm_(&trans_A, &trans_tensor, &m, &n, &k, &alpha, A,
             &m, T->data, &tensor_stride, &beta, C, &output_stride);
    } else {
      dgemm_layout = CblasColMajor;
      // trans_tensor = CblasNoTrans;
      trans_tensor = 'N';
      // trans_A = CblasTrans;
      trans_A = 'T';
      output_stride = left_dims_product;
      m = left_dims_product;
      // n = r;
      k = right_dims_product;
      i_r = r;
      tensor_stride = left_dims_product;
      dgemm_(&trans_tensor, &trans_A, &m, &i_r, &k, &alpha, T->data,
             &tensor_stride, A, &i_r, &beta, C, &output_stride);
    }
  }
  /**
          cblas_dgemm();
          1) dgemm_layout, controls the layout of the output matrix
          2) trans_tensor, balances dgemm_layout with contracting over left or
     right modes 3) trans_A, balanes dgemm_layout with the assumption A is
     always in row major ordering 4) m, rows of T->data 5) r, columns of A 6) k,
     common dimensions being contracted over 7) alpha 8) T->data, tensor data 9)
     k, stride of T->data 10) A, a row major matrix 11) r, stride of A 12) beta
          13) C, output matrix
          14) output_stride
          print_dgemm_inputs( dgemm_layout, trans_tensor, trans_A, m, r, k,
     alpha, k, r, beta, output_stride );
  */
  // cblas_dgemm(dgemm_layout, trans_tensor, trans_A, m, r, k, alpha, T->data,
  //             tensor_stride, A, r, beta, C, output_stride);
  // dgemm_(char *transa, char *transb, integer *m, integer *
  // n, integer *k, doublereal *alpha, doublereal *a, integer *lda,
  // doublereal *b, integer *ldb, doublereal *beta, doublereal *c, integer
  // *ldc)
  //  dgemm_(&trans_tensor, &trans_A, &m, &i_r, &k, &alpha, T->data,
  //  &tensor_stride,
  //         A, &i_r, &beta, C, &output_stride);
}

/**
        Wrapper function for partial_MTTKRP()
        Forms a desired KRP, manages all the memory for the KRP
        Passes the KRP as the argument double * A for partial_MTTKRP()
        1) output format
        2) s
        3) D
        4) Y, ktensor
        5) T, data tensor
        6) C, output matrix
        7) num_threads
*/
void partial_MTTKRP_with_KRP(Output_Layout OL, long int s, direction D,
                             ktensor *Y, tensor *T, double *C,
                             long int num_threads) {
  // mkl_set_num_threads(num_threads);
  // openblas_set_num_threads(num_threads);

  if ((s < 0) || (s >= T->nmodes - 1)) {  // s cannot be negative or be equal to
                                          // N-1 or be greater than N-1
    printf("Invalid value of s in partial_MTTKRP_with_KRP(), s = %d\nExit\n",
           s);
    exit(-1);
  }
  if (T == NULL) {
    printf("T == NULL in partial_MTTKRP_with_KRP()\nExit\n");
    exit(-1);
  }
  if (Y == NULL) {
    printf("Y == NULL in partial_MTTKRP_with_KRP()\nExit\n");
    exit(-1);
  }

  long int i, right_dims_product, left_dims_product, free_KRP;
  double *KRP;
  ktensor tempY;

  right_dims_product = 1;
  left_dims_product = 1;  // the left dimension product always includes s

  for (i = 0; i < s + 1; i++) left_dims_product *= T->dims[i];
  for (i = s + 1; i < T->nmodes; i++) right_dims_product *= T->dims[i];

  if (D == ::direction::left) {  // s+1 because left includes s by convention
    LR_Ktensor_Reordering_newY(Y, &tempY, s + 1, D);
  } else {
    LR_Ktensor_Reordering_newY(Y, &tempY, s, D);
  }

  if (tempY.nmodes == 1) {  // no KRP needs to be formed
    KRP = tempY.factors[0];
    free_KRP = 0;
  } else {
    KRP = reinterpret_cast<double *>(
        malloc(sizeof(double) * Y->rank * tempY.dims_product));
    wrapper_Parallel_Multi_revKRP(&tempY, num_threads, KRP);
    destruct_Ktensor(&tempY, 0);
    free_KRP = 1;
  }

  /**
          partial_MTTKRP();
          1) OL, specified output layuout
          2) s, dimension being split over
          3) Y->rank, number of columns in the KRP
          4) T, data tensor
          5) KRP, matrix to multiply into the data tensor
          6) C, output matrix C
          7) num_threads
  */
  partial_MTTKRP(OL, s, D, T, KRP, Y->rank, C, num_threads);

  if (free_KRP == 1) free(KRP);
}

/**
        partial_MTTKRP_with_KRP_output_FM();
        Wrapper function for computing a partial_MTTKRP with KRP and outputing a
   (Row_Major) factor matrix. This function should always be used on edges that
   lead to leaves of the tree. D tells you s

        1) direction D
        2) tensor T
        3) ktensor K, the full ktensor
        4) num_threads
*/
void partial_MTTKRP_with_KRP_output_FM(direction D, ktensor *Y, tensor *T,
                                       long int num_threads) {
  long int s, n;

  if (D == ::direction::left) {
    s = T->nmodes - 2;
    n = s + 1;
  } else {  // D == right
    s = 0;
    n = s;
  }

  /**
          1) RowMajor, output is a factor matrix
          2) s, determined above based on hte direction
          3) Y, k tensor corresponding to the original data tensor
          4) T, the original data tensor, this is a PM
          5) Y->factors[n], output to the factor matrix
          6) num_threads
  */
  partial_MTTKRP_with_KRP(RowMajor, s, D, Y, T, Y->factors[n], num_threads);
}

/**
        partial_MTTKRP_with_KRP_output_T()
        Wrapper function for computing partial_MTTKRP with KRP and outputing a
   tensor. This function should be used on edges of the tree that lead to
   non-leaf nodes. 1) s, split point 2) D, direction 3) input_ktensor, ktnesor
   whose factor matrices to are used to form the KRP 4) input_tensor, tensor to
   contract 5) output_tensor 6) num_threads
*/
void partial_MTTKRP_with_KRP_output_T(long int s, direction D,
                                      ktensor *input_ktensor,
                                      tensor *input_tensor,
                                      tensor *output_tensor,
                                      long int num_threads) {
  Output_Layout output_layout = ColMajor;
  long int new_s;  // new_s is the split point for the new tensor

  partial_MTTKRP_with_KRP(output_layout, s, D, input_ktensor, input_tensor,
                          output_tensor->data, num_threads);

  if (D == ::direction::left) {
    new_s = s;
  } else {
    new_s = s + 1;  // the LR_tensor_Reduction does not include the given split
  }

  // Adjust the tensor in the opposite direction of D
  LR_tensor_Reduction(input_tensor, output_tensor, new_s,
                      opposite_direction(D));

  // Ddd in the rank dimension
  output_tensor->nmodes += 1;
  output_tensor->dims[output_tensor->nmodes - 1] = input_ktensor->rank;
  output_tensor->dims_product *= input_ktensor->rank;
}

/**
        multi_TTV();
        performs a multi_ttv with between subtensors of T->data and the column
   of the matrix A. Outputs to the matrix/tensor C. 1) OL, output_Layout 2) D,
   direction 3) s, the split point 4) T, tensor 5) A, ttv the columns of this
   matrix with the tensor 6) r, columns in A 7) C, output matrix 8) num_threads
*/
void multi_TTV(Output_Layout OL, long int s, direction D, tensor *T, double *A,
               long int r, double *C, long int num_threads) {
  if (s < 0 || s >= T->nmodes - 1) {  // s cannot be negative or be equal to N-1
                                      // or be greater than N-1
    printf("Invalid value of s in multi_TTV(), s = %d\nExit\n", s);
    exit(-1);
  }
  if (T == NULL) {
    printf("T == NULL in multi_TTV()\nExit\n");
    exit(-1);
  }
  if (A == NULL) {
    printf("A == NULL in multi_TTV()\nExit\n");
    exit(-1);
  }
  if (C == NULL) {
    printf("C == NULL in multi_TTV()\nExit\n");
    exit(-1);
  }
  if (r != T->dims[T->nmodes - 1]) {  // the last mode of T must be of length r
    printf(
        "r and the last mode of the tensor T must be equal, r = %d, "
        "T->dims[T->nmodes-1] = %d\nmulti_TTV()\n\nExit\n",
        r, T->dims[T->nmodes - 1]);
    exit(-1);
  }

  CBLAS_ORDER dgemv_layout;
  CBLAS_TRANSPOSE trans_tensor;
  long int right_dims_product, left_dims_product, i, m, n, tensor_stride,
      output_stride, output_col_stride;
  double alpha, beta;

  alpha = 1.0;
  beta = 0.0;

  left_dims_product = 1;
  right_dims_product = 1;

  for (i = 0; i < s + 1; i++)  // by convenction left product includes s
    left_dims_product *= T->dims[i];
  for (i = s + 1; i < T->nmodes - 1;
       i++)  // -1 because we do not want to include the rank dimension
    right_dims_product *= T->dims[i];

  trans_tensor = CblasNoTrans;  // alwyas NoTrans because dgemv_layout is based
                                // on left and right
  if (D == ::direction::left) {
    dgemv_layout = CblasRowMajor;
    m = right_dims_product;
    n = left_dims_product;
    tensor_stride = n;
    if (OL == RowMajor) {
      output_stride = r;
      output_col_stride = 1;
    } else {
      output_stride = 1;
      output_col_stride = m;
    }
  } else {  // D == right
    dgemv_layout = CblasColMajor;
    m = left_dims_product;
    n = right_dims_product;
    tensor_stride = m;
    if (OL == RowMajor) {
      output_stride = r;
      output_col_stride = 1;
    } else {
      output_stride = 1;
      output_col_stride = m;
    }
  }

  // openblas_set_num_threads(num_threads);
  for (i = 0; i < r; i++) {
    /**
            1) dgemv_layout
            2) trans_tensor
            3) m, rows of T->data chunk
            4) n, cols of T->data chunk
            5) alpha
            6) T->data + i * m * n, i * size of chunk
            7) tensor_stride, stride of a tensor chunk
            8) A[i], start of the ith column of a
            9) r, stride of A
            10) beta
            11) C + i * output_col_stride, starting point of the ith col of C
            12) output_stride, distance between elements of the output vector
            print_dgemv_inputs( dgemv_layout, trans_tensor, m, n, alpha, i*m*n,
       tensor_stride, i, r, beta, output_col_stride, output_stride );
    */
    cblas_dgemv(dgemv_layout, trans_tensor, m, n, alpha, T->data + i * m * n,
                tensor_stride, A + i, r, beta, &C[i * output_col_stride],
                output_stride);
  }
}

/**
        multi_TTV_with_KRP();
        KRP wrapper for the general multi_TTV function.
        Forms a desired KRP and passes it to the multi_TTv function, manages all
   memory related to the KRP.
*/
void multi_TTV_with_KRP(Output_Layout OL, long int s, direction D, tensor *T,
                        ktensor *Y, double *C, long int num_threads) {
  ktensor tempY;
  double *KRP;
  long int free_KRP;

  // the tensour should have 1 more mode than the ktensor
  if (T->nmodes != Y->nmodes + 1) {
    printf(
        "In multi_TTV_with_KRP(), T->nmodes = %d and Y->nmodes = %d\nThe "
        "tensor should have 1 more mode than the ktensor.\nExit\n",
        T->nmodes, Y->nmodes);
    exit(-1);
  }

  // form the needed tempY, multi ttv goes the same way as PM now
  if (D == ::direction::left) {
    LR_Ktensor_Reordering_newY(Y, &tempY, s + 1, D);
  } else {  // D == right
    LR_Ktensor_Reordering_newY(Y, &tempY, s, D);
  }

  if (tempY.nmodes == 1) {
    KRP = tempY.factors[0];
    free_KRP = 0;
  } else {
    KRP = reinterpret_cast<double *>(
        malloc(sizeof(double) * Y->rank * tempY.dims_product));

    wrapper_Parallel_Multi_revKRP(&tempY, num_threads, KRP);
    destruct_Ktensor(&tempY, 0);
    free_KRP = 1;
  }

  multi_TTV(OL, s, D, T, KRP, Y->rank, C, num_threads);

  if (free_KRP == 1) free(KRP);
}

/**
        multi_TTV_with_KRP_output_FM();
        Wrapper function for performing a multi_TTV with a KRP and outputing a
   (RowMajor) factor matrix. This should be used on edges leading to leave of
   the tree. 1) D, direction to contract 2) input_tensor, tensor to contract 3)
   input_ktensor, factor matrices from which to form the KRP 4) num_threads
*/
void multi_TTV_with_KRP_output_FM(direction D, tensor *input_tensor,
                                  ktensor *input_ktensor,
                                  long int num_threads) {
  long int s, n;

  if (D == ::direction::left) {
    s = input_ktensor->nmodes - 2;
    n = s + 1;
  } else {  // D == right
    s = 0;
    n = s;
  }

  multi_TTV_with_KRP(RowMajor, s, D, input_tensor, input_ktensor,
                     input_ktensor->factors[n], num_threads);
}

void multi_TTV_with_KRP_output_T(long int s, direction D, tensor *input_tensor,
                                 ktensor *input_ktensor, tensor *output_tensor,
                                 long int num_threads) {
  long int x;
  direction op_D;

  multi_TTV_with_KRP(ColMajor, s, D, input_tensor, input_ktensor,
                     output_tensor->data, num_threads);

  if (D == ::direction::left) {
    op_D = ::direction::right;
    x = s;
  } else {  // D == right
    op_D = ::direction::left;
    x = s + 1;
  }

  // adjust the tensor in the op_D
  LR_tensor_Reduction(input_tensor, output_tensor, x, op_D);

  if (op_D ==
      ::direction::left) {  // if we went left we didn't get the rank dimension
    output_tensor->nmodes += 1;
    output_tensor->dims[output_tensor->nmodes - 1] = input_ktensor->rank;
    output_tensor->dims_product *= input_ktensor->rank;
  }
}

#endif  // DIMTREE_DIMTREES_HPP_
