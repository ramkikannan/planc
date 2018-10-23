/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNTF_DISTNTFIO_HPP_
#define DISTNTF_DISTNTFIO_HPP_

#include <unistd.h>
#include <armadillo>
#include <limits>  // for limits of standard data types
#include <string>
#include <vector>
#include "common/distutils.hpp"
#include "common/ncpfactors.hpp"
#include "common/npyio.hpp"
#include "common/tensor.hpp"
#include "distntf/distntfmpicomm.hpp"

/*
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
class DistNTFIO {
 private:
  const NTFMPICommunicator &m_mpicomm;
  Tensor m_A;
  // don't start getting prime number from 2;
  static const int kPrimeOffset = 10;
  // Hope no one hits on this number.
  static const int kW_seed_idx = 1210873;
  static const int kalpha = 1;
  static const int kbeta = 0;

  /*
   * Uses the pattern from the input matrix X but
   * the value is computed as low rank.
   */
  void randomLowRank(const UVEC i_global_dims, const UWORD i_k) {
    // start with the same seed_idx with the global dimensions
    // on all the MPI processor.
    NCPFactors global_factors(i_global_dims, i_k, false);
    global_factors.randu(kW_seed_idx);
    global_factors.normalize();
    int tensor_modes = global_factors.modes();
    // UVEC local_dims = i_global_dims / this->m_mpicomm.proc_grids();
    // determine local tensor sizes and start idxs
    UVEC start_row = arma::zeros<UVEC>(tensor_modes);
    UVEC local_dims_uvec = arma::zeros<UVEC>(tensor_modes);
    UVEC tmp_proc_grids = this->m_mpicomm.proc_grids();
    for (int i = 0; i < tensor_modes; i++) {
      local_dims_uvec[i] = itersplit(i_global_dims[i], tmp_proc_grids[i],
                                     this->m_mpicomm.fiber_rank(i));
      start_row[i] = startidx(i_global_dims[i], tmp_proc_grids[i],
                              this->m_mpicomm.fiber_rank(i));
    }
    NCPFactors local_factors(local_dims_uvec, i_k, false);
    UWORD end_row;
    for (int i = 0; i < local_factors.modes(); i++) {
      end_row = start_row[i] + local_dims_uvec(i) - 1;
      local_factors.factor(i) =
          global_factors.factor(i).rows(start_row[i], end_row);
    }
    local_factors.rankk_tensor(m_A);
  }

  void build_local_tensor() {
    // assumption is that the m_A has global tensor with it now.
    // Local_size = global_size/proc_grids // mode wise division. Local_size is
    // a vector of size modes For i over modes // at the end of this loop we
    // will have the start_idx and end_idx of every  mode for every rank
    // Start_idx(i) = MPI_FIBER_RANK(i) * local_dims(i)
    // End_idx(i) = start_idx(i) + local_size(i) - 1
    // List of subscripts = {set of subs for every mode} .
    // Find the Cartesian product of the list of idxs.
    // This will give us the list of tensor global subscripts
    // and should be in the row - major order.
    // Call the sub2ind of the tensor for all the global
    // subscripts and extract those data from the global data.

    // Local_size = global_size/proc_grids
    UVEC global_dims = m_A.dimensions();
    UVEC local_dims = global_dims / this->m_mpicomm.proc_grids();
    size_t num_modes = m_A.modes();
    std::vector<std::vector<size_t>> idxs_for_mode;
    size_t current_start_idx, current_end_idx, num_idxs_for_mode;
    // For i over modes
    for (size_t i = 0; i < num_modes; i++) {
      // Start_idx(i) = MPI_FIBER_RANK(i) * global_dims(i)
      current_start_idx = MPI_FIBER_RANK(i) * local_dims(i);
      // current_end_idx = current_start_idx + local_dims(i);
      // num_idxs_for_mode = current_end_idx -  current_start_idx;
      // UVEC idxs_current_mode = arma::linspace<UVEC>(current_start_idx,
      //                          current_end_idx - 1,
      //                          num_idxs_for_mode);
      std::vector<size_t> idxs_current_mode(local_dims(i));
      std::iota(std::begin(idxs_current_mode), std::end(idxs_current_mode),
                current_start_idx);
      // arma::conv_to<std::vector<size_t>>::from(idxs_current_mode);
      idxs_for_mode.push_back(idxs_current_mode);
    }
    // we have to now determing the cartesian product of this set.
    std::vector<std::vector<size_t>> global_idxs =
        cartesian_product(idxs_for_mode);
    Tensor local_tensor(local_dims);
    UVEC global_idxs_uvec = arma::zeros<UVEC>(global_idxs[0].size());
    for (size_t i = 0; i < global_idxs.size(); i++) {
      global_idxs_uvec = arma::conv_to<UVEC>::from(global_idxs[i]);
      local_tensor.m_data[i] = m_A.at(global_idxs_uvec);
    }
    this->m_A = local_tensor;
  }

 public:
  explicit DistNTFIO(const NTFMPICommunicator &mpic) : m_mpicomm(mpic) {}
  ~DistNTFIO() {
    // delete this->m_A;
  }
  void readInput(const std::string file_name) {
    // In this case we are reading from a file.
    // We can read .bin file, .npy file, .txt file.
    // Look at the extension and appropriately
    // .bin files must be supported by .info
    std::string extension = file_name.substr(file_name.find_last_of(".") + 1);
    std::stringstream ss;
    ss << file_name;
    if (MPI_SIZE > 1) {
      ss << "." << MPI_RANK;
      DISTPRINTINFO("Reading input file::" << ss.str());
    }
    if (extension == "bin") {
      // make sure you have corresponding info file
      // in txt format. First line specifier number of modes
      // second line specifies dimensions of each mode seperated
      // by space
      std::ios_base::openmode mode = std::ios_base::in | std::ios_base::binary;
      m_A.read(ss.str(), mode);
    } else if (extension == "npy") {
      NumPyArray npyreader;
      npyreader.load(file_name);
      if (MPI_RANK == 0) {
        npyreader.printInfo();
      }
      this->m_A = *(npyreader.m_input_tensor);
      if (MPI_SIZE > 1) {
        build_local_tensor();
      }
    } else if (extension == "txt") {
      m_A.read(ss.str());
    }
  }

  /*
      Reading from real input file.
      Expecting a .tensor text file and .bin file.
      .info text file every one will read
      .bin file will be an mpi io file
      Read distributed tensor
  */
  UVEC read_dist_tensor(const std::string filename,
                        UVEC *start_idxs_uvec = NULL) {
    // all processes reading the file_name.info file.
    std::string filename_no_extension =
        filename.substr(0, filename.find_last_of("."));
    filename_no_extension.append(".info");
    std::ifstream ifs;
    int modes;
    // info file always in text mode
    ifs.open(filename_no_extension, std::ios_base::in);
    // write modes
    ifs >> modes;
    // dimension of modes
    // UVEC global_dims = arma::zeros<UVEC>(this->m_modes);
    int *global_dims = new int[modes];
    int *local_dims = new int[modes];
    int *start_idxs = new int[modes];
    UVEC local_dims_uvec = arma::zeros<UVEC>(modes);
    UVEC global_dims_uvec = arma::zeros<UVEC>(modes);
    UVEC tmp_proc_grids = this->m_mpicomm.proc_grids();
    for (int i = 0; i < modes; i++) {
      ifs >> global_dims[i];
      global_dims_uvec[i] = global_dims[i];
      local_dims[i] = itersplit(global_dims[i], tmp_proc_grids[i],
                                this->m_mpicomm.fiber_rank(i));
      local_dims_uvec[i] = local_dims[i];
      start_idxs[i] = startidx(global_dims[i], tmp_proc_grids[i],
                               this->m_mpicomm.fiber_rank(i));
      if (start_idxs_uvec != NULL) start_idxs_uvec[i] = start_idxs[i];
    }
    ifs.close();
    Tensor rc(local_dims_uvec);
    // Create the datatype associated with this layout
    MPI_Datatype view;
    MPI_Type_create_subarray(modes, global_dims, local_dims, start_idxs,
                             MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
    MPI_Type_commit(&view);
    // Open the file
    MPI_File fh;
    // const MPI_Comm &comm = Y->getDistribution()->getComm(true);
    // get confirmed with grey
    int ret = MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY,
                            MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
      DISTPRINTINFO("Error: Could not read file " << filename << std::endl);
    }
    // Set the view
    MPI_Offset disp = 0;
    MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);
    // Read the file
    int count = rc.numel();
    assert(count <= std::numeric_limits<int>::max());
    if (ISROOT && 8 * count > std::numeric_limits<int>::max()) {
      PRINTROOT("file read size ::" << 8 * count << " > 2GB" << std::endl);
    }
    MPI_Status status;
    ret = MPI_File_read_all(fh, rc.m_data, count, MPI_DOUBLE, &status);
    int nread;
    MPI_Get_count(&status, MPI_DOUBLE, &nread);
    if (ret != MPI_SUCCESS) {
      DISTPRINTINFO("Error: Could not read file " << filename << std::endl);
    }
    // Close the file
    MPI_File_close(&fh);
    // Free the datatype
    MPI_Type_free(&view);

    // free the allocated things.
    delete[] global_dims;
    delete[] local_dims;
    delete[] start_idxs;
    this->m_A = rc;
    rc.clear();
    for (int i = 0; i < MPI_SIZE; i++) {
      if (i == MPI_RANK) {
        DISTPRINTINFO("local tensor:")
        this->m_A.print();
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    return global_dims_uvec;
  }
  /*
    Reading from real input file.
    Expecting a .tensor text file and .bin file.
    .info text file will be written by root processor alone
    .bin file will be an mpi io file
    Read distributed tensor
*/
  void write_dist_tensor(const std::string filename,
                         const Tensor &local_tensor) {
    // all processes reading the file_name.info file.
    std::string filename_no_extension =
        filename.substr(0, filename.find_last_of("."));
    int modes = local_tensor.modes();
    int *gsizes = new int[modes];
    int *lsizes = new int[modes];
    int *starts = new int[modes];
    UVEC tmp_proc_grids = this->m_mpicomm.proc_grids();
    for (int i = 0; i < modes; i++) {
      lsizes[i] = local_tensor.dimension(i);
      MPI_Allreduce(&lsizes[i], &gsizes[i], 1, MPI_INT, MPI_SUM,
                    this->m_mpicomm.fiber(i));
      starts[i] =
          startidx(gsizes[i], tmp_proc_grids[i], this->m_mpicomm.fiber_rank(i));
    }
    // info file always in text mode
    if (ISROOT) {
      filename_no_extension.append(".info");
      std::ofstream ofs;
      ofs.open(filename_no_extension, std::ios_base::out | std::ios::app);
      // write modes
      ofs << modes << std::endl;
      // dimension of modes
      for (int i = 0; i < modes; i++) {
        ofs << gsizes[i] << std::endl;
      }
      ofs.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Now each of write to the file.
    // Create the datatype associated with this layout

    MPI_Datatype view;
    MPI_Type_create_subarray(modes, gsizes, lsizes, starts, MPI_ORDER_FORTRAN,
                             MPI_DOUBLE, &view);
    MPI_Type_commit(&view);

    // Open the file
    MPI_File fh;
    int ret =
        MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (ISROOT && ret != MPI_SUCCESS) {
      DISTPRINTINFO("Error: Could not open file " << filename << std::endl);
    }

    // Set the view
    MPI_Offset disp = 0;
    MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

    // Write the file
    int count = local_tensor.numel();
    assert(count <= std::numeric_limits<int>::max());
    MPI_Status status;
    ret =
        MPI_File_write_all(fh, local_tensor.m_data, count, MPI_DOUBLE, &status);
    if (ret != MPI_SUCCESS) {
      DISTPRINTINFO("Error: Could not write file " << filename << std::endl);
    }
    // Close the file
    MPI_File_close(&fh);
    // Free the datatype
    MPI_Type_free(&view);
    // free the allocated memories
    delete[] gsizes;
    delete[] lsizes;
    delete[] starts;
  }

  /*
   * We need m,n,pr,pc only for rand matrices. If otherwise we are
   * expecting the file will hold all the details.
   * If we are loading by file name we dont need distio flag.
   *
   */
  void readInput(const std::string file_name, UVEC i_global_dims,
                 UVEC i_proc_grids, UWORD k = 0, double sparsity = 0) {
    // INFO << "readInput::" << file_name << "::" << distio << "::"
    //     << m << "::" << n << "::" << pr << "::" << pc
    //     << "::" << this->MPI_RANK << "::" << this->m_mpicomm.size() <<
    //     std::endl;
    std::string rand_prefix("rand_");
    if (!file_name.compare(0, rand_prefix.size(), rand_prefix)) {
      // random generators
      // determine local dimensions size
      if (!file_name.compare("rand_uniform")) {
        UVEC tmp_proc_grids = this->m_mpicomm.proc_grids();
        int modes = i_global_dims.n_elem;
        UVEC local_dims_uvec = arma::zeros<UVEC>(modes);
        for (int i = 0; i < modes; i++) {
          local_dims_uvec[i] = itersplit(i_global_dims[i], tmp_proc_grids[i],
                                         this->m_mpicomm.fiber_rank(i));
        }
        Tensor temp(local_dims_uvec);
        this->m_A = temp;
        // generate again. otherwise all processes will have
        // same input tensor because of the same seed.
        this->m_A.randu(449 * MPI_RANK + 677);
      } else if (!file_name.compare("rand_lowrank")) {
        randomLowRank(i_global_dims, k);
      }
    } else {
      read_dist_tensor(file_name);
    }
  }
  void write(const NCPFactors &factors, const std::string &output_file_name) {
    for (int i = 0; i < factors.modes(); i++) {
      std::stringstream sw;
      sw << output_file_name << "_mode" << i << "_" << MPI_SIZE << "_"
         << MPI_RANK;
      factors.factor(i).save(sw.str(), arma::raw_ascii);
    }
  }
  void writeRandInput() {}
  const Tensor &A() const { return m_A; }
  const NTFMPICommunicator &mpicomm() const { return m_mpicomm; }
};
}  // namespace planc

#endif  // DISTNTF_DISTNTFIO_HPP_
