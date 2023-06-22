/* Copyright 2016 Ramakrishnan Kannan */
#ifndef COMMON_DISTUTILS_HPP_
#define COMMON_DISTUTILS_HPP_

#include <mpi.h>
#include <string>
#include "common/distutils.h"
#include "common/utils.h"
#include "common/utils.hpp"

using namespace std;

struct procInfo {
  // processor coordinates
  int i;
  int j;
  // data indices
  int start_idx;
  int nrows;      // end_idx = start_idx + nrows
};

struct gridInfo {
  // grid dimensions
  int pr;
  int pc;
  // 1D ordering
  char order;
};

inline void mpitic() {
  // tictoc_stack.push(clock());
  tictoc_stack.push(std::chrono::steady_clock::now());
}

inline void mpitic(int rank) {
  // std::cout << "tic::" << rank << "::" << std::chrono::steady_clock::now() <<
  // std::endl;
  tictoc_stack.push(std::chrono::steady_clock::now());
}

inline double mpitoc(int rank) {
#ifdef __WITH__BARRIER__TIMING__
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - tictoc_stack.top());
  double rc = time_span.count();
  tictoc_stack.pop();
  std::cout << "toc::" << rank << "::" << rc << std::endl;
  return rc;
}

inline double mpitoc() {
#ifdef __WITH__BARRIER__TIMING__
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - tictoc_stack.top());
  double rc = time_span.count();
  tictoc_stack.pop();
  return rc;
}

int mpi_serial_print(function<void()> p){//string p){
  MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);
    // TODO: wrap this in function so we can reuse it whilst debugging...
    // Helpful code for printing things out in order that is useful for debugging:
    // https://stackoverflow.com/questions/17570996/mpi-printing-in-an-order#:~:text=There%20is%20no%20way%20for,in%20whatever%20order%20you%20like.
    int rank;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int message = rank;
    MPI_Request request;
    if (rank == 0) {INFO << "==============================" << std::endl;}
    if(size==1){
      printf("1 SIZE = %d RANK = %d MESSAGE = %d \n",size,rank, message);
      p();
    } else if (rank == 0) {
        printf("1 SIZE = %d RANK = %d MESSAGE = %d \n",size,rank, message);
        // INFO << p << endl;
        p();
        MPI_Isend(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    } else {
        int buffer;
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &buffer);
        if (buffer == 1) {
            printf("2 SIZE = %d RANK = %d MESSAGE = %d \n",size,rank, message);
            // INFO << p << endl;
            p();
            MPI_Irecv(&message, buffer, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
            if (rank + 1 != size) {
                MPI_Isend(&message, 1, MPI_INT, ++rank, 0, MPI_COMM_WORLD, &request);
            }
        };
    };
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

/**
 * Captures the memory usage of every mpi process
 */
inline void memusage(const int myrank, std::string event) {
  // Based on the answer from stackoverflow
  // http://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-run-time-in-c
  int64_t rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL) return;
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return;
  }
  fclose(fp);
  int64_t current_proc_mem = (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
  // INFO << myrank << "::mem::" << current_proc_mem << std::endl;
  int64_t allprocmem;
  MPI_Reduce(&current_proc_mem, &allprocmem, 1, MPI_INT64_T, MPI_SUM, 0,
             MPI_COMM_WORLD);
  if (myrank == 0) {
    INFO << event << " total rss::" << allprocmem << std::endl;
  }
}

/**
 * The dimension a particular rank holds out of 
 * the global dimension n across p processes.
 * @param[in] n is the global size of the tensor on that dimension
 * @param[in] p is the number of splits of n. 
 *            Typically number of processes on a particular mode
 * @param[in] r is the rank of the mpi process in that mode. fiber rank
 */
inline int itersplit(int n, int p, int r) {
  int split = (r < n % p) ? n / p + 1 : n / p;
  return split;
}

/**
 * Returns the start idx of the current rank r
 * for a global dimension n across p processes. 
 * @param[in] n is the size of the tensor on that dimension
 * @param[in] p is the number of splits of n.
 * @param[in] r is the rank of the mpi process in that mode. fiber rank
 */
inline int startidx(int n, int p, int r) {
  int rem = n % p;
  int idx =
      (r < rem) ? r * (n / p + 1) : (rem * (n / p + 1) + ((r - rem) * (n / p)));
  return idx;
}

/**
 * Returns the linear/flattened index of an element in a pr x pc grid,
 * given the subscript index,
 * in either row-major/column-major ordering.
 * @param[in] x is the row-rank of the element in the grid
 * @param[in] y is the column-rank of the element in the grid
 * @param[in] pr is the number of rows in the grid
 * @param[in] pc is the number of columns in the grid
 * @param[in] order row-major ('R') or column-major ('C')
 * @param[out] idx is the linear/flattened index
 */
inline int sub2ind(int x, int y, int pr, int pc, char order='C') {
  int idx = -1;
  if (order == 'C') {
    idx = (y * pr) + x;
  } else {
    idx = (x * pc) + y;
  }
  return idx;
}

/**
 * Returns the subscripts of an element in a pr x pc grid,
 * given a linear/flattened index,
 * in either row-major/column-major ordering.
 * @param[in] idx is the linear index of the element in the grid
 * @param[in] pr is the number of rows in the grid
 * @param[in] pc is the number of columns in the grid
 * @param[in] order row-major ('R') or column-major ('C')
 * @param[out] coords is the subscripts of the element in the grid
 */
inline std::vector<int> ind2sub(int idx, int pr, int pc, char order='C') {
  std::vector<int> coords = {-1, -1};
  if (order == 'C') {
    coords[0] = idx % pr;
    coords[1] = idx / pr;
  } else {
    coords[0] = idx / pc;
    coords[1] = idx % pc;
  }
  return coords;
}

/**
 * Returns the p2p communication rank given a MPI cartesian grid
 * and coordinates.
 * @param[in] gridComm MPI communicator with Cartesian structure
 * @param[in] coords Processor coordinates in the logical grid
 * @param[out] rank p2p communication rank of the given coordinates
 */
int mpisub2ind(MPI_Comm gridComm, std::vector<int> coords) {
  int rank = -1;
  MPI_Cart_rank(gridComm, &coords[0], &rank);
  return rank;
}

/**
 * Returns the next element to (i, j) in a pr x pc grid.
 * Requires (i, j) to not be the last processor (pr-1, pc-1).
 * @param[in] i current processor's x-coordinate
 * @param[in] j current processor's y-coordinate
 * @param[in] pr grid x-dimension size
 * @param[in] pc grid y-dimension size
 * @param[in] order processor ordering the grid
 * @param[out] nproc (i, j) coordinates to the next processor
 */
std::vector<int> nextproc(int i, int j, int pr, int pc, char order='C') {
  assert((i != pr-1) || (j != pc-1));
  if (order == 'C') {
    if (i == pr-1) {
      i = 0;
      j++;
    } else {
      i++;
    }
  } else {
    if (j == pc-1) {
      j = 0;
      i++;
    } else {
      j++;
    }
  }
  std::vector<int> nproc = {i, j};
  return nproc;
}

/**
 * Get the global row range of the 1D distributed factor matrices
 * in a 2D grid of size pr x pc.
 * @param[in] n is the size of the factor matrix (n x k)
 * @param[in] i is the processor's first coordinate
 * @param[in] j is the processor's second coordinate
 * @param[in] pr is the grid's first dimension
 * @param[in] pc is the grid's second dimension
 * @param[in] ord is the row/major ordering of the 1D distribution
 * @param[out] pinfo is the processor information
 */
procInfo getrownum(int n, int i, int j, int pr, int pc, char ord='C') {
  int global_n, global_row_idx, local_row_idx, local_nrows;
  procInfo pinfo;

  // Check based on the ordering
  if (ord == 'C') {
    global_n       = itersplit(n, pc, j);
    global_row_idx = startidx(n, pc, j);

    local_nrows   = itersplit(global_n, pr, i);
    local_row_idx = global_row_idx + startidx(global_n, pr, i);
  } else {
    global_n       = itersplit(n, pr, i);
    global_row_idx = startidx(n, pr, i);

    local_nrows   = itersplit(global_n, pc, j);
    local_row_idx = global_row_idx + startidx(global_n, pc, j);
  }
  // Set the structure values
  pinfo.i = i;
  pinfo.j = j;
  pinfo.start_idx = local_row_idx;
  pinfo.nrows = local_nrows;

  return pinfo;
}

/**
 * Get the processor which contains the row in the processor grid.
 * @param[in] n is the size of the factor matrix (n x k)
 * @param[in] sidx is the row index of the factor matrix
 * @param[in] pr is the grid's first dimension
 * @param[in] pc is the grid's second dimension
 * @param[in] ord is the row/major ordering of the 1D distribution
 * @param[out] pinfo is the processor information
 */
procInfo getfirstproc(int n, int sidx, int pr, int pc, char ord='C') {
  procInfo pinfo;
  bool found = false;

  // Check based on the ordering
  if (ord == 'C') {
    for (int jj = 0; jj < pc; jj++) {
      for (int ii = 0; ii < pr; ii++) {
        pinfo = getrownum(n, ii, jj, pr, pc, ord);
        if (sidx >= pinfo.start_idx 
              && sidx < (pinfo.start_idx + pinfo.nrows)) {
          found = true;
          break;
        }
      }
      if (found) break;
    }
  } else {
    for (int ii = 0; ii < pr; ii++) {
      for (int jj = 0; jj < pc; jj++) {
        pinfo = getrownum(n, ii, jj, pr, pc, ord);
        if (sidx >= pinfo.start_idx 
              && sidx < (pinfo.start_idx + pinfo.nrows)) {
          found = true;
          break;
        }
      }
      if (found) break;
    }
  }

  return pinfo;
}

/**
 * Get the p2p communications needed to map two different 1D distributions
 * across two processor grids A (r1 x c1) and B (r2 x c2).
 * The factor matrices can be distributed in row-major or column-major
 * ordering. We always assume that messages are sent from the A grid to
 * the B grid. In the sending case (i, j) is the processor index in A grid.
 * @param[in] n is the number of the rows of the 1D distributed factor matrix
 * @param[in] i is the processor coordinate ((i, j) in A grid)
 * @param[in] j is the processor coordinate ((i, j) in A grid)
 * @param[in] A is the grid (with the sending processor (i, j))
 * @param[in] B is the grid (with the receiving processors)
 * @param[out] tosend vector containing the processor ids (in B grid) to send
 *                    to and offset lengths (in rows)       
 */
std::vector<procInfo> getsendinfo(int n, int i, int j,
                          gridInfo A, gridInfo B) {
  std::vector<procInfo> tosend;

  // Get your factor's row numbers (in A grid)
  procInfo localp = getrownum(n, i, j, A.pr, A.pc, A.order);

  // Get the corresponding processor (in B grid)
  procInfo firstp = getfirstproc(n, localp.start_idx, B.pr, B.pc, B.order);

  // Prepare the first message
  int endidx  = firstp.start_idx + firstp.nrows;
  int maxsend = std::min(endidx - localp.start_idx, localp.nrows);
  
  procInfo msg1;
  msg1.i         = firstp.i;
  msg1.j         = firstp.j;
  msg1.start_idx = 0;
  msg1.nrows     = maxsend;

  tosend.push_back(msg1);

  // Check if another message is needed. We only need 1 extra send.
  if (maxsend < localp.nrows) {
    std::vector<int> nextpcoords = nextproc(firstp.i, firstp.j, 
                                                B.pr, B.pc, B.order);
    procInfo msg2;
    msg2.i         = nextpcoords[0];
    msg2.j         = nextpcoords[1];
    msg2.start_idx = maxsend;
    msg2.nrows     = localp.nrows - maxsend;

    tosend.push_back(msg2);
  }

  return tosend;
}

/**
 * Get the p2p communications needed to map two different 1D distributions
 * across two processor grids A (r1 x c1) and B (r2 x c2).
 * The factor matrices can be distributed in row-major or column-major
 * ordering. We always assume that messages are sent from the A grid to
 * the B grid. In the receiving case (i, j) is the processor index in B grid.
 * @param[in] n is the number of the rows of the 1D distributed factor matrix
 * @param[in] i is the processor coordinate ((i, j) in B grid)
 * @param[in] j is the processor coordinate ((i, j) in B grid)
 * @param[in] A is the grid (with the sending processors)
 * @param[in] B is the grid (with the receiving processor (i, j))
 * @param[out] torecv vector containing the processor ids (in A grid) to
 *                    receive from and offset lengths (in rows)       
 */
std::vector<procInfo> getrecvinfo(int n, int i, int j,
                          gridInfo A, gridInfo B) {
  std::vector<procInfo> torecv;

  // Get your factor's row numbers (in B grid)
  procInfo localp = getrownum(n, i, j, B.pr, B.pc, B.order);

  // Get the corresponding processor (in A grid)
  procInfo firstp = getfirstproc(n, localp.start_idx, A.pr, A.pc, A.order);

  // Prepare the first message
  int endidx  = firstp.start_idx + firstp.nrows;
  int maxrecv = std::min(endidx - localp.start_idx, localp.nrows);
  
  procInfo msg1;
  msg1.i         = firstp.i;
  msg1.j         = firstp.j;
  msg1.start_idx = 0;
  msg1.nrows     = maxrecv;

  torecv.push_back(msg1);

  // Check if another message is needed. We only need 1 extra recv.
  if (maxrecv < localp.nrows) {
    std::vector<int> nextpcoords = nextproc(firstp.i, firstp.j, 
                                                A.pr, A.pc, A.order);
    procInfo msg2;
    msg2.i         = nextpcoords[0];
    msg2.j         = nextpcoords[1];
    msg2.start_idx = maxrecv;
    msg2.nrows     = localp.nrows - maxrecv;

    torecv.push_back(msg2);
  }

  return torecv;
}

template<class Type>
int ss_tok(stringstream& ss, Type& arg){

  ss >> arg;

  // cout << arg << endl;

  return 0;
}

// NOTE: do we want to use variadic templates here instead?...
// int ss_tok_int(string line, initializer_list<int>& nums){
template<typename Type, typename... Types>
int ss_tok(stringstream& ss, Type& arg, Types&... args){

  ss >> arg;

  // cout << arg << ", ";

  // NOTE: how can we pop off the first entry of line when we recurse?... ----> by passing in
  // referenced stringstream it will hopefully keep its state...
  ss_tok(ss, args...);

  return 0;
}

// Convert string into variable number of mixed-type variables 
template<typename... Types>
int ss_tok(string line, Types&... args){
  stringstream ss;
  ss.str(line);
  ss_tok(ss, args...);

  return 0;
}

// SPARSE
// NOTE: this function, as all io functions in PLANC, assumes all the entries are zero indexed. this
// function reads in txt files that just contain entries in sparse format (row, col, val) with no
// meta information. currently need to preprocess .mtx files to enforce this, but can add in
// function to explicitly read in .mtx files in the future. note that as of now we assume the user
// passes in the dimension information via the input arguments, which is why we do not need the meta
// information. we should probably add in error checking to make sure all indices of read in matrix
// are with the dimensions passed in by the user.
void readInputUnpartitionedMatrix(UWORD m, UWORD n, 
                      const std::string& input_file_name, 
                      SP_MAT &m_A, const MPI_Comm gridComm, 
                      int unpartitioned) {

  int i = 1;
  while(i < 1){}
  std::vector<int> dimSizes;
  std::vector<int> periods;
  std::vector<int> coords;

  int pr, pc, rank, row_rank, col_rank;
  
  // Get the p2p rank
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Get the grid details
  int nd = 2;
  dimSizes.resize(nd);
  periods.resize(nd);
  coords.resize(nd);
  MPI_Cart_get(gridComm, nd, &dimSizes[0], &periods[0], &(coords[0]));

  pr = dimSizes[0];
  pc = dimSizes[1];
  row_rank = coords[0];
  col_rank = coords[1];


  int srow = startidx(m, pr, row_rank); //itersplit(m, pr, row_rank);
  int scol = startidx(n, pc, col_rank); //itersplit(n, pc, col_rank);
  int num_row =  itersplit(m, pr, row_rank);
  int num_col = itersplit(n, pc, col_rank);
  // cout << "rank: " << rank  << " srow: " << srow << " scol: " << scol << endl;
  ifstream fin(input_file_name);
  int row, col;
  double val; //NOTE: assume double precision
  string test;
  stringstream ss;
  // while(!fin.eof()){

  // check if .mtx or .txt file
  // string::iterator it = (*input_file_name).end();
  string suffix = input_file_name.substr(input_file_name.size() - 4);
  cout << "SUFFIX: " << suffix << endl;
  // while(true){};
  int shift = 0;

  if(suffix == ".mtx"){
    shift = 1; 
    // discard first three lines
    for(int i = 0; i < 3; i++){
      getline(fin, test);
    }
  }

  // TODO: figure out how to pass initializer_list values by reference so they change...
  // initializer_list<int> coords_tuple = {row, col, val};
  switch(unpartitioned){

    // naive method that requires reading in the file twice...
    case 1:
    {
      // get number non-zero
      int nnz = 0;
      while(getline(fin, test)){
        // cout << "test: " << test << endl;
        ss_tok(test, row, col, val); //coords_tuple);
        
        // if we read in mtx file, then convert to 0 indexing...
        row = row - shift;
        col = col - shift;

        // cout << "row: " << row << " col: " << col << " val: " << val << endl;
        if( ((srow < row + 1)  && (row  < srow + num_row)) && ((scol < col + 1)  && (col  < scol + num_col)) ){
          // NOTE: it might be most efficient to iterate through file to determine how many elements to
          // add to the matrix, then allocate the matrix, and then reread through the file and add
          // elements to it instead of modify the matrix size at every iteration... ----> use umat
          // locations and vec values example from documentation to insert all values at once instead of
          // reading in the file twice... ----> put indices and values in array, then use that to
          // initialize umat?... ----> is that possible?...
          nnz++;
        }
      }

      if(0<nnz){
    // read in entries into umat and vec, then batch allocate local sp_mat
        arma::umat indx(2, nnz);
        arma::vec vals(nnz);

        fin.clear();
        fin.seekg(0);
        int count = 0;


        if(suffix == ".mtx"){
          shift = 1; 
          // discard first three lines
          for(int i = 0; i < 3; i++){
            getline(fin, test);
          }
        }

        while(getline(fin, test)){
          // cout << "test: " << test << endl;
          ss_tok(test, row, col, val); //coords_tuple);

          // if we read in mtx file, then convert to 0 indexing...
          row = row - shift;
          col = col - shift;

          // cout << "row: " << row << " col: " << col << " val: " << val << endl;
          if( ((srow < row + 1)  && (row  < srow + num_row)) && ((scol < col + 1)  && (col  < scol + num_col)) ){
            indx(0, count) = row - srow;
            indx(1, count) = col - scol;
            vals(count) = val;
            // cout << indx(0, count) << ", " << indx(1, count) << ", " << vals(count) << endl;
            count++;
          }
        }
        // cout << "after file" << endl;
        fin.close();
        // cout << "indx: " << endl;
        // indx.print();
        // cout << "vals: " << endl;
        // vals.print();
        SP_MAT temp_spmat(indx, vals);
        m_A = temp_spmat;
      }else{
        arma::umat idxs = arma::zeros<arma::umat>(2, 1);
        VEC vals = arma::zeros<VEC>(1);
        SP_MAT temp_spmat(idxs, vals);
        m_A = temp_spmat;
      }


      m_A.resize(num_row, num_col);
      break;
    }

    // all processes read in entire file and append to vectors that store coordinate information...
    case 2:
    {
      while(getline(fin, test)){
        ss_tok(test, row, col, val); //coords_tuple);
      }
      break;
    }
  }

  // auto ma_print = [&m_A](){
  //   m_A.print();
  // };
  // mpi_serial_print(ma_print);
  // m_A.print(); //brief_
  // while(true){}
}

// SPARSE
void readInputMatrix(UWORD m, UWORD n, 
                      const std::string& input_file_name, 
                      SP_MAT &m_A, const MPI_Comm gridComm, int unpartitioned = 1) {
  if(unpartitioned==0){
    std::stringstream sr;
    std::vector<int> dimSizes;
    std::vector<int> periods;
    std::vector<int> coords;

    int pr, pc, rank, row_rank, col_rank;
    
    // Get the p2p rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Get the grid details
    int nd = 2;
    dimSizes.resize(nd);
    periods.resize(nd);
    coords.resize(nd);
    MPI_Cart_get(gridComm, nd, &dimSizes[0], &periods[0], &(coords[0]));

    pr = dimSizes[0];
    pc = dimSizes[1];
    row_rank = coords[0];
    col_rank = coords[1];

    sr << input_file_name << row_rank << "_" << col_rank;

    INFO << "sparse readInputMatrix: " << sr.str() << std::endl;

    int srow = itersplit(m, pr, row_rank);
    int scol = itersplit(n, pc, col_rank);

    MAT temp_ijv;
    temp_ijv.load(sr.str(), arma::raw_ascii);
    if (temp_ijv.n_rows > 0 && temp_ijv.n_cols > 0) {
      MAT vals(2, temp_ijv.n_rows);
      MAT idxs_only = temp_ijv.cols(0, 1);
      arma::umat idxs = arma::conv_to<arma::umat>::from(idxs_only);
      arma::umat idxst = idxs.t();
      vals = temp_ijv.col(2);
      SP_MAT temp_spmat(idxst, vals);
      m_A = temp_spmat;
    } else {
      arma::umat idxs = arma::zeros<arma::umat>(2, 1);
      VEC vals = arma::zeros<VEC>(1);
      SP_MAT temp_spmat(idxs, vals);
      m_A = temp_spmat;
    }
    // m_A.load(sr.str(), arma::coord_ascii);
    // uniform_dist_matrix(m_A);
    m_A.resize(srow, scol);
  }else{
    readInputUnpartitionedMatrix(m, n, input_file_name, m_A, gridComm, unpartitioned);
  }
}

// DENSE
// NOTE: adding the partitioned variable is a temporary fix. A more permanent solution is to pass in
// templated structs so that the function signatures are all exactly the same. In this case if we
// wanted to add in a parameter we just add it to the struct...
void readInputMatrix(UWORD global_m, UWORD global_n,
                      const std::string& input_file_name,
                      MAT &m_A, const MPI_Comm gridComm, int unpartitioned = 1) {
  INFO << "dense readInputMatrix: " << input_file_name << std::endl;

  std::vector<int> dimSizes;
  std::vector<int> periods;
  std::vector<int> coords;

  int pr, pc, rank, row_rank, col_rank;
  
  // Get the p2p rank
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Get the grid details
  int nd = 2;
  dimSizes.resize(nd);
  periods.resize(nd);
  coords.resize(nd);
  MPI_Cart_get(gridComm, nd, &dimSizes[0], &periods[0], &(coords[0]));

  pr = dimSizes[0];
  pc = dimSizes[1];
  row_rank = coords[0];
  col_rank = coords[1];

  int local_m = itersplit(global_m, pr, row_rank);
  int local_n = itersplit(global_n, pc, col_rank);
  m_A.zeros(local_m, local_n);

  int gsizes[] = {global_m, global_n};
  int lsizes[] = {local_m, local_n};
  int starts[] = {startidx(global_m, pr, row_rank),
                  startidx(global_n, pc, col_rank)};
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Datatype view;
  MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_FORTRAN,
                            MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  MPI_File fh;
  int ret = MPI_File_open(MPI_COMM_WORLD, input_file_name.c_str(),
                          MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if ((rank == 0) && ret != MPI_SUCCESS) {
    INFO << rank << "::" << __PRETTY_FUNCTION__ << "::" << __LINE__          
      << "::" << std::endl << "Error: Could not open file " 
      << input_file_name << std::endl << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Offset disp = 0;
  MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

  int count = lsizes[0] * lsizes[1];
  MPI_Status status;
  ret = MPI_File_read_all(fh, m_A.memptr(), count, MPI_DOUBLE, &status);
  if ((rank == 0) && ret != MPI_SUCCESS) {
    INFO << rank << "::" << __PRETTY_FUNCTION__ << "::" << __LINE__          
      << "::" << std::endl << "Error: Could not read file " 
      << input_file_name << std::endl << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_File_close(&fh);
  MPI_Type_free(&view);
}

template<typename MATTYPE>
int print_mat(MATTYPE X){
  X.print();
  return 0;
}

#endif  // COMMON_DISTUTILS_HPP_
