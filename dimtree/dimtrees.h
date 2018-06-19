/* Copyright Koby Hayashi 2018 */

#ifndef DIMTREE_DIMTREES_H_
#define DIMTREE_DIMTREES_H_

#include "dimtree/kdttensor.h"

/**
    This .h file contains all the functions need to construct dimension
    trees for fast_CP_ALS algorithms.
*/

/**
    Enumerator for denoting desired output formats
*/
typedef enum { RowMajor, ColMajor } Output_Layout;

/**
    Partial MTTKRP functions
*/

void partial_MTTKRP(Output_Layout OL, long int s, direction D, tensor *T,
                    double *A, long int r, double *C, long int num_threads);
void partial_MTTKRP_with_KRP(Output_Layout OL, long int s, direction D,
                             ktensor *Y, tensor *T, double *C,
                             long int num_threads);
void partial_MTTKRP_with_KRP_output_FM(direction D, ktensor *Y, tensor *T,
                                       long int num_threads);
void partial_MTTKRP_with_KRP_output_T(long int s, direction D,
                                      ktensor *input_ktensor,
                                      tensor *input_tensor,
                                      tensor *output_tensor,
                                      long int num_threads);

void multi_TTV(Output_Layout OL, long int s, direction D, tensor *T, double *A,
               long int r, double *C, long int num_threads);
void multi_TTV_with_KRP(Output_Layout OL, long int s, direction D, tensor *T,
                        ktensor *Y, double *C, long int num_threads);
void multi_TTV_with_KRP_output_FM(direction D, tensor *input_tensor,
                                  ktensor *input_ktensor, long int num_threads);
void multi_TTV_with_KRP_output_T(long int s, direction D, tensor *input_tensor,
                                 ktensor *input_ktensor, tensor *output_tensor,
                                 long int num_threads);
#endif // DIMTREE_DIMTREES_H_
