/* Copyright 2022 Ramakrishnan Kannan */

#ifndef DISTJNMF_DISTJNMFIO_HPP_
#define DISTJNMF_DISTJNMFIO_HPP_

#include <unistd.h>

#include <armadillo>
#include <cinttypes>
#include <string>

#include <algorithm>
#include <random>

#include "common/utils.hpp"
#include "common/distutils.hpp"
#include "distjointnmf/jnmfmpicomm.hpp"
#include <set>

/**
 * File name formats
 * A is the filename
 * 1D distribution Arows_totalpartitions_rank or Acols_totalpartitions_rank
 * Double 1D distribution (both row and col distributed)
 * Arows_totalpartitions_rank and Acols_totalpartitions_rank
 * TWOD distribution A_totalpartition_rank
 * Just send the first parameter Arows and the second parameter Acols to be
 * zero.
 */

namespace planc {

template <class MATTYPE>
class DistJointNMFIO {
 private:
  const MPICommunicatorJNMF& m_mpicomm;
  MATTYPE& m_A;  /// TWOD
  // don't start getting prime number from 2;
  static const int kPrimeOffset = 10;
  // Hope no one hits on this number.
  static const int kW_seed_idx = WTRUE_SEED;
  static const int kH_seed_idx = HTRUE_SEED;
  static const int k_row_seed = 160669;
  static const int k_col_seed = 162287;
  static const int k_seed_off = 617237;
#ifdef BUILD_SPARSE
  static const int kalpha = 5;
  static const int kbeta = 10;
#else
  static const int kalpha = 1;
  static const int kbeta = 0;
#endif
  IVEC rcounts;  // vector to hold row counts for uneven splits
  IVEC ccounts;  // vector to hold column counts for uneven splits


// TODO: change this to just TWOD ----> delete everthing in this file that isn;t for TWOD case, 1D,
// etc... ----> use generate_rand_matrix for uniform and normal cases ----> the low-rank and
// symmetric cases are trickier...
  // const iodistributions m_distio;
  /**
   * A random matrix is always needed for sparse case
   * to get the pattern. That is., the indices where
   * numbers are being filled. We will use this pattern
   * and change the number later.
   */
  // TODO: template this function (or overload) to get rid of preprocessor macros ----> ended up
  // overloading
  void randMatrix(const std::string type, const int primeseedidx,
                  double sparsity, const int symm,
                  const bool adj_rand, SP_MAT* X,
                  UWORD m = 0, UWORD n = 0, UWORD k = 0,
                  bool lowrank = false) {

    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();

    int n_rows = (*X).n_rows;
    int n_cols = (*X).n_cols;

    DISTPRINTINFO("sparse randMatrix::" 
                << primeseedidx << "::sp=" << sparsity);

    // TODO: fix this to handle four cases: sym, sym p=1, non-sym, non-sym p=1
    if (symm && (m_mpicomm.size() > 1)) {
      if (lowrank) {
        gen_rand_symm_rect(X, primeseedidx, "uniform", n, 10, sparsity);    
        randomLowRank(m, n, k, symm, adj_rand, X); //NOTE: does this function work here?...
      } else { //TODO: not differentiating between uniform and normal for now...
        gen_rand_symm_rect(X, primeseedidx, type, n, 10, sparsity);
      }
    } else if (symm && (m_mpicomm.size() == 1)) {
      // Call the shared memory function
      generate_rand_matrix(*X, type, m, n, k, sparsity, symm, adj_rand);
    } else {
      if (type == "uniform") { //NOTE: does this generate low-rank?...
        (*X).sprandu((*X).n_rows, (*X).n_cols, sparsity);
      } else if (type == "normal") {
        (*X).sprandn((*X).n_rows, (*X).n_cols, sparsity);
      } else if (type == "lowrank") {
        (*X).sprandu((*X).n_rows, (*X).n_cols, sparsity);
        randomLowRank(m, n, k, symm, adj_rand, X);  
      }
    }

#ifdef APPEND_TIME
auto start = chrono::steady_clock::now();
// mpitic();
#endif
    // NOTE: should this part be inside or outside the else statement?...
    SP_MAT::iterator start_it = (*X).begin();
    SP_MAT::iterator end_it = (*X).end();
    if (adj_rand) {
      for (SP_MAT::iterator it = start_it; it != end_it; ++it) {
        double currentValue = (*it);
        (*it) = ceil(kalpha * currentValue + kbeta);
        if ((*it) < 0) (*it) = kbeta;
      }
    } else {
      // Set the negative values to positive
      for (SP_MAT::iterator it = start_it; it != end_it; ++it) {
        if ((*it) < 0) (*it) = std::abs(*it);
      }
    }
#ifdef APPEND_TIME
auto end = chrono::steady_clock::now();
chrono::duration<double> elapsed_seconds = end-start;
if(m_mpicomm.rank() == 0){printf("- post-processing took %.3lf secs.\n", elapsed_seconds.count());}
#endif
  }

  void randMatrix(const std::string type, const int primeseedidx,
                  double sparsity, const int symm,
                  const bool adj_rand, MAT* X,
                  UWORD m = 0, UWORD n = 0, UWORD k =0,
                  bool lowrank = false) {

    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();

    int n_rows = (*X).n_rows;
    int n_cols = (*X).n_cols;

    DISTPRINTINFO("dense randMatrix::" << primeseedidx);

    // TODO: fix this to handle four cases: sym, sym p=1, non-sym, non-sym p=1
    if (symm && (m_mpicomm.size() > 1)) {
      if (lowrank) { // low-rank
        gen_rand_lowrank_symm_rect(X, primeseedidx, type, n, k);
      } else {
        gen_rand_symm_rect(X, primeseedidx, type, n, 10);
      }
    } else if (symm && (m_mpicomm.size() == 1)) {
      // Call the shared memory function
      generate_rand_matrix(*X, type, m, n, k, sparsity, symm, adj_rand);
    } else {
      if (type == "uniform") {
        X->randu();
      } else if (type == "normal") {
        X->randn();
      } else if(type == "lowrank"){
        randomLowRank(m, n, k, symm, adj_rand, X);
      }
    }

    if (adj_rand) {
      (*X) = kalpha * (*X) + kbeta;
      (*X) = ceil(*X);
    }
    (*X).elem(find((*X) < 0)).zeros();
  }

  void gen_rand_lowrank_symm_rect(MAT* X, const int primeseedidx, const std::string type, UWORD n, UWORD k){
    INFO << this->m_mpicomm.rank() << "::Hello from gen_rand_lowrank_symm_rect" << endl;
    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();
    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();
    int count_pr = n % pr;
    int count_pc = n % pc;
    int n_rows = X->n_rows;
    int n_cols = X->n_cols;

    MAT H_1(n_rows, k);
    MAT H_2(k, n_cols);

    int row_start = ((count_pr < row_rank) ? (row_rank - count_pr) * (n/pr) + count_pr * (n/pr + 1)
                                            :  row_rank * (n/pr + 1));

    int col_start = ((count_pc < col_rank) ? (col_rank - count_pc) * (n/pc) + count_pc * (n/pc + 1)
                                            :  col_rank * (n/pc + 1));

    gen_discard(row_start, n_rows, k, H_1, false, kH_seed_idx);

    gen_discard(col_start, n_cols, k, H_2, true, kH_seed_idx);

    auto f_X = [this, H_1, H_2, X](){ //[temp, H_1, H_2, X](){ //
      INFO << "(" << m_mpicomm.row_rank() << ", " << m_mpicomm.col_rank() << ")" << endl;
      INFO << "H_1: " << endl;
      H_1.print();
      INFO << "H_2: " << endl;
      H_2.print();
      INFO << "X: " << endl;
      X->print();
      };
    *X = (H_1 * H_2);

    INFO << this->m_mpicomm.rank() << "::Leaving gen_rand_lowrank_symm_rect" << endl;
  }


  template<typename t, typename tt>
  double cut(t gen_value, tt cutoff_value,  const std::string type){
    // INFO << "gen_value: " << gen_value << endl;
    return (gen_value < cutoff_value) ? gen_value : 0;
  }

  // NOTE: I bet we could generalize this to accept a variable number of mersenne twisters, but
  // could we generalize it to any rng with a seed function?...
  int reseed(long reseed_value, mt19937& jump_rand, mt19937& num_rand){
    // cout << "RESEEDING WITH VALUE: " << reseed_value << endl;
    jump_rand.seed(reseed_value);
    num_rand.seed(reseed_value);
    return 0;
  }

  // TODO: document this...
  void gen_rand_symm_rect(SP_MAT*& X, const int primeseedidx, const std::string type, UWORD n, int max_val, float sparsity = 1){
    int i = 1;
    while(i < 1){}
#ifdef APPEND_TIME
tic();
#endif

    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();
    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();
    int count_pr = n % pr;
    int count_pc = n % pc;
    int n_rows = X->n_rows;
    int n_cols = X->n_cols;

    double temp;
    auto gen_val = [this, &temp, type] (std::mt19937 &gen) {
      temp = gen();
      if (type == "uniform") {
          // making between 0 and 1...
          temp = temp / gen.max();
          return 1.0;
      } else if (type == "normal") {
          std::mt19937 gen_norm(temp);
          std::normal_distribution<double> dist(0, 1);
          temp = abs(dist(gen_norm));
          return 1.0;
      }
      return 0.0;
    };

    std::pair<int, int> corner_index( startidx(n, pr, row_rank), startidx(n, pc, col_rank));

    //compute reseed array
    // int* ra = new int[pr + pc];
    std::vector<int> ra(pr + pc);

    for(int i = 0; i < pr; ++i){ra[i] = startidx(n, pr, i);}
    for(int i = 0; i < pc; ++i){ra[i + pr] = startidx(n, pc, i);}
   

    std::set<int> s(ra.begin(), ra.end());
    ra.assign(s.begin(), s.end());
    ra.push_back(n);
    int unique = ra.size();

    // INFO << "RESEED ARRAY: " << endl;
    // for(int i = 0; i < unique; ++i){
    //   INFO << ra[i] << ", ";
    // }INFO << endl;


    double jump_mod = 2/sparsity; // + 1;
    int jump, entry, debug;
    double append_time = 0;

    std::vector<int> row_idxs, col_idxs;
    std::vector<double> vals;
    
    int rai;
    //NOTE: should we group these together into a pair?...
    std::mt19937 jump_rand;
    std::mt19937 num_rand;
    for(int h = 0; h < 2; ++h){
      // go across rows ----> instead of copy, pasting, and modifying we will just swap n_rows
      // with n_cols and corner_index.first with corner_index.second ----> that way we can
      // avoid copy-paste errors and only need to debug one code block...
      if(h == 1){
        swap(n_rows, n_cols);
        swap(corner_index.first, corner_index.second);
      }

      int orig_rai = std::distance(ra.begin(), std::find(ra.begin(), ra.end(), corner_index.first));
      // NOTE: these columns are independent, could potentially make this outer loop parallel to speed
      // things up in the case of running with hybrid parallelism
      for(int i = 0; i < n_cols; ++i){
        // go back to starting row after switching columns
        rai = orig_rai;
        //NOTE: corner_index.first is contained in the reseed array...
        reseed(corner_index.second + i + ra[rai]*n, jump_rand, num_rand);

        if((corner_index.second + i) < corner_index.first){ // this column completely below diagonal
          // jump = fmod(jump_rand(), jump_mod); first row/col of each local matrix is a reseed
          // point...
          for(int j = 0; j < n_rows; j = j + jump){

            // jumped out of current seed range, reseeding...
            if(corner_index.first + j >= ra[rai + 1]){ 
              rai++;
              reseed(corner_index.second + i + ra[rai]*n, jump_rand, num_rand);
              j = ra[rai] - corner_index.first; // jumped out of range returning to reseed index...
              jump = 0;
              continue;
            }

            jump = std::floor(std::fmod(jump_rand(), jump_mod));
            //TODO: instead should append to three vectors and then batch allocate
            //TODO: implement option for either uniform or normal
            debug = gen_val(num_rand); //num_rand() % 10;
            // cout << "CURRENT RESEED ARRAY VALUE: "<< ra[rai] << endl;
#ifdef APPEND_TIME
tic();
#endif
            if(jump>0){
              if(h == 0){
                // X->at(j, i) = debug;
                row_idxs.push_back(j);
                col_idxs.push_back(i);
              }else{
                // X->at(i, j) = debug;
                row_idxs.push_back(i);
                col_idxs.push_back(j);
              }
              vals.push_back(temp);
            }
#ifdef APPEND_TIME
append_time += toc();
#endif
          } 
        }else if((corner_index.second + i) < (corner_index.first + n_rows)){ // this column contains diagonal

          //NOTE: do we need to subtract one here?... 
          // ----> yes, but check for case when less than 1...
          rai = std::distance(ra.begin(), std::lower_bound(ra.begin(), ra.end(), corner_index.second + i)) - 1;
          if(rai < 0){
            rai = 0;
          }
          reseed(corner_index.second + i + ra[rai]*n, jump_rand, num_rand);

          // jump = fmod(jump_rand(), jump_mod);
          // entry = num_rand();

          for(int j = ra[rai]-corner_index.first; j < n_rows; j = j + jump){ //start from reseed point..

            // jump within starting range until we pass the diagonal
            if((corner_index.second + i) > (corner_index.first + j)  ){ // error here... starting in middle of col or start of row ----> inconsistent... >=
              jump = std::floor(std::fmod(jump_rand(), jump_mod));
              entry = num_rand() % 10;
            }else{
              if((corner_index.first + j) >= ra[rai + 1]){ // jumped out of current range, reseeding
                rai++;
                // NOTE: should the reseed here also use the row as well?...
                reseed(corner_index.second + i + ra[rai]*n, jump_rand, num_rand);
                j = ra[rai] - corner_index.first; //// what if j >n_rows?...
                jump = 0;
                continue;
              }

              jump = std::floor(std::fmod(jump_rand(), jump_mod));
              debug = gen_val(num_rand); //num_rand() % 10;
              // cout << "CURRENT RESEED ARRAY VALUE: "<< ra[rai] << endl;
#ifdef APPEND_TIME
tic();
#endif
              if(jump>0){
                if(h == 0){
                  // X->at(j, i) = debug;
                  row_idxs.push_back(j);
                  col_idxs.push_back(i);
                  vals.push_back(temp);
                }else{
                  // X->at(i, j) = debug;
                  if((corner_index.second + i) != (corner_index.first + j)){
                    row_idxs.push_back(i);
                    col_idxs.push_back(j);
                    vals.push_back(temp);
                  }
                }
              }
#ifdef APPEND_TIME
append_time += toc();
#endif
            }
          }
        }else{
          // this column completely above diagonal, nothing to do here...
          // cout << "column completely above diagonal, nothing to do here..." << endl;
        }
      }
    }
    //switch them back
    swap(n_rows, n_cols);
    swap(corner_index.first, corner_index.second);

#ifdef APPEND_TIME
tic();
#endif
    //batch allocate
    int nnz = vals.size();
    arma::umat test_locations(2, nnz);
    arma::vec test_values(nnz);
    // arma::vec values(vals);
    
    // there is probably a more efficient way of doing this...
    // does accessing the three vectors within the same loop negatively impact performance?... ---->
    // we can probably rely upon the compiler to optimize this...
    for(int i = 0; i < nnz; ++i){//std::vector<uint>::iterator it)
      test_locations(0, i) = row_idxs[i];
      test_locations(1, i) = col_idxs[i];
      test_values(i) = vals[i]; 
    }
    // SP_MAT temp_spmat(true, test_locations, test_values, n_rows, n_cols); //temp_spmat;
    // *X = std::move(temp_spmat);//new SP_MAT(true, test_locations, test_values, n_rows, n_cols); //temp_spmat;

    *X = SP_MAT(true, test_locations, test_values, n_rows, n_cols);

#ifdef APPEND_TIME
double allocate_time = toc();
#endif

#ifdef APPEND_TIME
double elapsed_seconds_gen_time = toc();
if(m_mpicomm.rank() == 0){
  printf("S append took %.3lf secs.\n", append_time);
  printf("S batch allocate took %.3lf secs.\n", allocate_time);
  printf("S gen besides appending and allocating took %.3lf secs.\n", elapsed_seconds_gen_time - append_time - allocate_time);
}
#endif 
  }


void gen_rand_symm_rect(MAT* X, const int primeseedidx, const std::string type,
        UWORD n, int max_val, float sparsity = 1) {

    int i = 1;
    while(i < 1){}

    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();
    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();
    int count_pr = n % pr;
    int count_pc = n % pc;
    int n_rows = X->n_rows;
    int n_cols = X->n_cols;

    pair<int, int> corner_index( startidx(n, pr, row_rank), 
                      startidx(n, pc, col_rank));
    
    int global_row_idx;
    int global_col_idx;

    /* Error function inverse is not a standard part of C math libraries.
     * Rolling a simple approximation for randn generation.
     * https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
     */

    double temp;
    auto gen_val = [this, &temp, type] (std::mt19937 &gen) {
      temp = gen();
      if (type == "uniform") {
          // making between 0 and 1...
          temp = temp / gen.max();
          return 1.0;
      } else if (type == "normal") {
          std::mt19937 gen_norm(temp);
          normal_distribution<double> dist(0, 1);
          temp = abs(dist(gen_norm));
          return 1.0;
      }
      return 0.0;
    };

    /*
     * TODO{seswar3}: Clean up code here. Column-major ordering does not
     * overcome the time to discard different seeds per row.
     */
    /*
    // Access matrix in column-major order
    for (int j = 0; j < n_cols; j++) {
      global_col_idx = corner_index.second + j;

      if (corner_index.first > global_col_idx) {
        // Column entirely below diagonal
        // Single seed for the column

        // Discard up to the starting row
        std::mt19937 gen(global_col_idx);
        gen.discard(corner_index.first - global_col_idx);
        
        // Fill in entries
        for (int i = 0; i < n_rows; i++) {
          if (gen_val(gen)) {
            X->at(i, j) =  temp;
          }
        }
      } else if ((corner_index.first + n_rows) < global_col_idx) {
        // Column entirely above diagonal
        // Multiple seeds (per row)

        for (int i = 0; i < n_rows; i++) {
          //Discard up to the starting column
          global_row_idx = corner_index.first + i;
          
          // Discard up to the starting row
          std::mt19937 gen(global_row_idx);
          gen.discard(global_col_idx - global_row_idx);
        
          // Fill in entry
          if (gen_val(gen)) {
            X->at(i, j) = temp;
          }
        }
      } else {
        // Column contains the diagonal
        // Fill in above-diagonal part
        for (int i = 0; i < global_col_idx - corner_index.first; i++) {
          // Discard up to the starting column
          global_row_idx = corner_index.first + i;
          
          // Discard up to the starting row
          std::mt19937 gen(global_row_idx);
          gen.discard(global_col_idx - global_row_idx);
        
          // Fill in entry
          if (gen_val(gen)) {
            X->at(i, j) = temp;
          }
        }
  
        // Fill in diagonal and below part
        std::mt19937 gen(global_col_idx);
      
        for (int i = global_col_idx - corner_index.first; i < n_rows; i++) {
          // Fill in entry
          if (gen_val(gen)) {
            X->at(i, j) = temp;
          }
        }
      }
    }
    */

    // Generate matrix in arrowhead fashion
    // seed with col-index
    for (int j = 0; j < n_cols; j++) {
      global_col_idx = corner_index.second + j;

      std::mt19937 gen(global_col_idx);
      gen.discard(corner_index.first);
      if (corner_index.first > global_col_idx) {
        // Column entirely below diagonal
        for (int i = 0; i < n_rows; i++) {
          if (gen_val(gen)) {
            X->at(i, j) =  temp;
          }
        }
      } else if ((corner_index.first <= global_col_idx) && 
          ((corner_index.first + n_rows) > corner_index.second)) {
        // Column contains the diagonal
        // Discard entries up to the diagonal
        int nrows_above_diag = global_col_idx - corner_index.first;
        gen.discard(nrows_above_diag);

        for (int i = nrows_above_diag; i < n_rows; i++) {
          if (gen_val(gen)) {
            X->at(i, j) =  temp;
          }
        }
      } else { 
        // Column entirely above diagonal, nothing to do here...
      }
    }

    // seed with row-index
    for (int i = 0; i < n_rows; i++) {
      global_row_idx = corner_index.first + i;
      
      std::mt19937 gen(global_row_idx);
      gen.discard(corner_index.second);
      if (corner_index.second > global_row_idx) { 
        // Row is entirely to the right of the diagonal
        for (int j = 0; j < n_cols; j++) {
          if (gen_val(gen)) {
            X->at(i, j) = temp;
          }
        }
      } else if ((corner_index.second <= global_row_idx) && 
          ((corner_index.second + n_cols) > corner_index.first)) { 
        // Row contains the diagonal
        // Discard entries up to and including the diagonal
        int ncols_left_diag = global_row_idx - corner_index.second + 1;
        gen.discard(ncols_left_diag);

        for (int j = ncols_left_diag; j < n_cols; j++) {
          if (gen_val(gen)) {
            X->at(i, j) = temp;
          }
        }
      } else { 
        // Row entirely to the left of the diagonal, nothing to do here...
      }
    }
#ifdef MPI_VERBOSE
    auto p = [this, X]() {
      cout << m_mpicomm.row_rank() << ", " << m_mpicomm.col_rank() << endl;
      X->print();
    };
    mpi_serial_print(p);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }


#ifndef BUILD_SPARSE
  /* normalization */
  // TODO: add in Laplacian normalization for square, symmetric connection matrices...
  void normalize(normtype i_normtype) {
    ROWVEC globalnormA = arma::zeros<ROWVEC>(m_A.n_cols);
    ROWVEC normc = arma::zeros<ROWVEC>(m_A.n_cols);
    MATTYPE normmat = arma::zeros<MATTYPE>(m_A.n_rows, m_A.n_cols);
      if (i_normtype == L2NORM) {
        normc = arma::sum(arma::square(m_A));
        MPI_Allreduce(normc.memptr(), globalnormA.memptr(), m_A.n_cols,
                      MPI_DOUBLE, MPI_SUM, this->m_mpicomm.commSubs()[1]);
      } else if (i_normtype == MAXNORM) {
        normc = arma::max(m_A);
        MPI_Allreduce(normc.memptr(), globalnormA.memptr(), m_A.n_cols,
                      MPI_DOUBLE, MPI_SUM, this->m_mpicomm.commSubs()[1]);
      }
    normmat = arma::repmat(globalnormA, m_A.n_rows, 1);
    m_A /= normmat;
  }
#endif

  /**
   * Uses the pattern from the input matrix X but
   * the value is computed as low rank.
   */
  // TODO: generating lowrank on rectangular grid will be tricky
  void randomLowRank(const UWORD m, const UWORD n, const UWORD k,
                     const int symm, const bool adj_rand, MAT* X) {
    uint start_row = 0, start_col = 0;
    uint nrows = 0, ncols = 0;

    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();

    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();

    start_row = startidx(m, pr, row_rank);
    nrows = (*X).n_rows;
    start_col = startidx(n, pc, col_rank);
    ncols = (*X).n_cols;
       
    // all machines will generate same Wrnd and Hrnd
    // at all times.
    MAT Wrnd(nrows, k);
    if (symm) {
      // seed W with H
      gen_discard(start_row, nrows, k, Wrnd, false, kH_seed_idx);
    } else {
      gen_discard(start_row, nrows, k, Wrnd, false, kW_seed_idx);
    }

    MAT Hrnd(k, ncols);
    gen_discard(start_col, ncols, k, Hrnd, true, kH_seed_idx);
    
    (*X) = Wrnd * Hrnd;
    if (adj_rand) {
      (*X).for_each(
          [](MAT::elem_type& val) { val = ceil(kalpha * val + kbeta); });
    }
  }

  /**
   * Uses the pattern from the input matrix X but
   * the value is computed as low rank.
   */
  void randomLowRank(const UWORD m, const UWORD n, const UWORD k,
                     const int symm, const bool adj_rand, SP_MAT* X) {
    uint start_row = 0, start_col = 0;
    uint end_row = 0, end_col = 0;

    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();

    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();

    start_row = startidx(m, pr, row_rank);
    end_row = (*X).n_rows;
    start_col = startidx(n, pc, col_rank);
    end_col = (*X).n_cols;
       
    // all machines will generate same Wrnd and Hrnd
    // at all times.
    MAT Wrnd((*X).n_rows, k);
    if (symm) {
      // seed W with H
      gen_discard(start_row, end_row, k, Wrnd, false, kH_seed_idx);
    } else {
      gen_discard(start_row, end_row, k, Wrnd, false, kW_seed_idx);
    }

    MAT Hrnd(k, (*X).n_cols);
    gen_discard(start_col, end_col, k, Hrnd, true, kH_seed_idx);

    SP_MAT::iterator start_it = (*X).begin();
    SP_MAT::iterator end_it = (*X).end();
    double tempVal = 0.0;
    for (SP_MAT::iterator it = start_it; it != end_it; ++it) {
      VEC Wrndi = vectorise(Wrnd.row(it.row()));
      VEC Hrndj = Hrnd.col(it.col());
      tempVal = dot(Wrndi, Hrndj);
      if (adj_rand) {
        (*it) = ceil(kalpha * tempVal + kbeta);
      } else {
        (*it) = tempVal;
      }
    }
  }

 public:
  DistJointNMFIO<MATTYPE>(const MPICommunicatorJNMF& mpic, MATTYPE& A)
      : m_mpicomm(mpic), m_A(A) {}

  /**
   * We need m,n,pr,pc only for rand matrices. If otherwise we are
   * expecting the file will hold all the details.
   * If we are loading by file name we dont need distio flag.
   * If the filename is rand_lowrank/rand_uniform, appropriate
   * random functions will be called. Otherwise, it will be loaded from file.
   * @param[in] file_name. For random matrices rand_lowrank/rand_uniform
   * @param[in] m - globalm. Needed only for random matrices.
   *                otherwise, we will know from file.
   * @param[in] n - globaln. Needed only for random matrices
   * @param[in] k - low rank. Used for generating synthetic lowrank matrices.
   * @param[in] sparsity - sparsity factor between 0-1 for sparse matrices.
   * @param[in] pr - Number of row processors in the 2D processor grid
   * @param[in] pc - Number of columnn processors in the 2D processor grid
   * @param[in] symm - Flag to create symmetric matrix
   * @param[in] adj_rand - Flag to run elementwise scaling on each entry.
   * @param[in] normalization - L2 column normalization of input matrix.
   */
  // TODO: remove all ifdef preprocessor macros ----> some existing function in common/utils.hpp
  // that might be useful, such as: read_input_matrix
  // template<class T>
  void readInput(const std::string file_name, UWORD m = 0, UWORD n = 0,
                 UWORD k = 0, double sparsity = 0, UWORD pr = 0, UWORD pc = 0,
                 const int symm = 0, bool adj_rand = false,
                 normtype i_normalization = NONE, int unpartitioned = 1) {
    // cout << "UNPARTITIONED: " << unpartitioned << endl;
    std::string rand_prefix("rand_");
    // Check file sizes
    if (symm) {
      assert(m == n);
    }

#ifdef APPEND_TIME
auto start = chrono::steady_clock::now();
#endif

    if (!file_name.compare(0, rand_prefix.size(), rand_prefix)) {
      std::string type = file_name.substr(rand_prefix.size());
      assert(type == "normal" || type == "lowrank" || type == "uniform");

      // Initialise the jagged splits
      rcounts.zeros(pr);
      ccounts.zeros(pc);

      for (int r = 0; r < pr; r++) rcounts[r] = itersplit(m, pr, r);

      for (int r = 0; r < pc; r++) ccounts[r] = itersplit(n, pc, r);

      int row_rank = m_mpicomm.row_rank();
      int col_rank = m_mpicomm.col_rank();

      m_A.zeros(rcounts[row_rank], ccounts[col_rank]);
      INFO << "calling rand matrix, symm: " << symm << std::endl;

      // TODO: generating lowrank on rectangular grid will be tricky
      bool lowrank = false;
      if (type == "lowrank") {
        lowrank = true;
      }
      

      const int temp = MPI_RANK + kPrimeOffset;

#ifdef APPEND_TIME
auto end = chrono::steady_clock::now();
chrono::duration<double> elapsed_seconds = end-start;
if(m_mpicomm.rank() == 0){printf("- pre-processing took %.3lf secs.\n", elapsed_seconds.count());}
#endif

      randMatrix(type, temp, sparsity, symm,
                  adj_rand, &m_A, m, n, k, lowrank);

    } else {

      readInputMatrix(m, n, file_name, m_A, m_mpicomm.gridComm(), unpartitioned);

    }
#ifndef BUILD_SPARSE
    if (i_normalization != NONE) {
      normalize(i_normalization);
    }
#endif

  }

  // TODO: move readInputMatrix common/distutils.hpp and overload based upon dense/sparse ---->
  // instead of referencing m_a as a memeber variable passs it as input to overload the function

  /**
   * Writes the factor matrix as output_file_name_W/H
   * @param[in] Local W factor matrix
   * @param[in] Local H factor matrix
   * @param[in] output file name
   */
  // TODO: take in a factor matrix and the distribution instead of just doing W and H by default
  // ----> need for writing out factor matrices such as H_hat
  void writeOutput(const MAT& W, const MAT& H,
                   const std::string& output_file_name) {
    std::stringstream sw, sh;
    sw << output_file_name << "_W";
    sh << output_file_name << "_H";

    INFO << sw.str() << std::endl;
    INFO << sh.str() << std::endl;

    int size = m_mpicomm.size();

    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();

    int rrank = m_mpicomm.row_rank();
    int crank = m_mpicomm.col_rank();

    int Wm = W.n_rows;
    int Hm = H.n_rows;

    // AllReduce global sizes of W,H in case W,H are inconsistent
    int global_Wm, global_Hm;
    MPI_Allreduce(&Wm, &global_Wm, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&Hm, &global_Hm, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int Widx = startidx(global_Wm, pr, rrank) +
               startidx(itersplit(global_Wm, pr, rrank), pc, crank);
    int Hidx = startidx(global_Hm, pc, crank) +
               startidx(itersplit(global_Hm, pc, crank), pr, rrank);

    writeOutputMatrix(W, global_Wm, Widx, sw.str());
    writeOutputMatrix(H, global_Hm, Hidx, sh.str());
  }

  void writeOutput(const MAT& output_mat, const std::string& output_file_name){
    int size = m_mpicomm.size();

    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();

    int rrank = m_mpicomm.row_rank();
    int crank = m_mpicomm.col_rank();

    int output_mat_m = output_mat.n_rows;
    int global_m;
    MPI_Allreduce(&output_mat_m, &global_m, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int Midx = startidx(global_m, pr, rrank) +
               startidx(itersplit(global_m, pr, rrank), pc, crank);

    writeOutputMatrix(output_mat, global_m, Midx, output_file_name);
  }

  /**
   * Uses MPIIO to write an output matrix X to a binary file with output_file_name.
   * X is assumed to be distributed in a 1D row-wise process grid.
   * @param[in] local output matrix
   * @param[in] global row count
   * @param[in] starting index in global matrix
   * @param[in] output file name
   */
  void writeOutputMatrix(const MAT& X, int global_m, int idx,
                         const std::string& output_file_name) {
    int rank = m_mpicomm.rank();
    int size = m_mpicomm.size();

    int local_m = X.n_rows;
    int global_n = X.n_cols;

    int gsizes[] = {global_m, global_n};
    int lsizes[] = {local_m, global_n};
    int starts[] = {idx, 0};

    MPI_Datatype view;
    if (local_m > 0) {  // process owns some part of X
      MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_FORTRAN,
                               MPI_DOUBLE, &view);
    } else {  // process owns no data and so previous function will fail
      MPI_Type_contiguous(0, MPI_DOUBLE, &view);
    }
    MPI_Type_commit(&view);

    MPI_File fh;
    int ret =
        MPI_File_open(m_mpicomm.gridComm(), output_file_name.c_str(),
                      MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    if (ISROOT & ret != MPI_SUCCESS) {
      DISTPRINTINFO("Error: Could not open file " << output_file_name
                                                  << std::endl);
    }

    MPI_Offset disp = 0;
    MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

    int count = lsizes[0] * lsizes[1];
    MPI_Status status;
    ret = MPI_File_write_all(fh, X.memptr(), count, MPI_DOUBLE, &status);
    if (ISROOT && ret != MPI_SUCCESS) {
      DISTPRINTINFO("Error could not write file " << output_file_name
                                                  << std::endl);
    }
    MPI_File_close(&fh);
    MPI_Type_free(&view);
  }

  // TODO: get rid of all 1D cases. Change to MPI_File_write_all for dense case ----> want local
  // sprase mat to be written out in pieces, whilst the dense case should be written out in a single
  // file ----> keep in mind that we only partition the input for the sparse case...
  // NOTE: look into reading/writing sparse matrices as a single (not partitioned file) using MPI
  // ----> in this case each proc would read in matrix, but only save the portion that it needs
  // ----> would definitely be more user friendly, but would it impact I/O performance?
  // ----> if it impacts I/O performace for large problem izes, could we still provide the file
  // partitioning option?
  void writeRandInput(string prefix, bool type) {
    std::stringstream sr, sc;
      sr << prefix << "_" << MPI_SIZE << "_" << MPI_RANK;
      DISTPRINTINFO("Writing rand input file " << sr.str()
                                               << PRINTMATINFO(m_A));

    if (!type) { //sparse
      this->m_A.save(sr.str(), arma::coord_ascii);
    } else { //dense
      this->m_A.save(sr.str(), arma::raw_ascii);
    }

  }
  const MATTYPE& A() const { return m_A; }
  const MPICommunicatorJNMF& mpicomm() const { return m_mpicomm; }
  const IVEC& row_counts() const { return rcounts; }
  const IVEC& col_counts() const { return ccounts; }
};

}  // namespace planc

// run with mpi run 3.
/*
void testDistJointNMFIO(char argc, char* argv[]) {
  planc::MPICommunicatorJNMF mpicomm(argc, argv);
#ifdef BUILD_SPARSE
  SP_MAT A;
  planc::DistJointNMFIO<SP_MAT> dio(mpicomm, ONED_DOUBLE, A);
#else
  MAT A;
  planc::DistJointNMFIO<MAT> dio(mpicomm, ONED_DOUBLE, A);
#endif
  dio.readInput("rand", 12, 9, 0.5);
  INFO << "Arows:" << mpicomm.rank() << std::endl
       << arma::conv_to<MAT>::from(dio.Arows()) << std::endl;
  INFO << "Acols:" << mpicomm.rank() << std::endl
       << arma::conv_to<MAT>::from(dio.Acols()) << std::endl;
}
*/
#endif  // DISTJNMF_DISTJNMFIO_HPP_
