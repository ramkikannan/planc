/* Copyright 2018 Ramakrishnan Kannan */
#ifndef COMMON_NPYIO_HPP_
#define COMMON_NPYIO_HPP_
#include <armadillo>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include "common/tensor.hpp"
#include "common/utils.h"

namespace planc {
class NumPyArray {
 private:
  int64_t m_word_size;
  bool m_fortran_order;
  int64_t m_modes;
  UVEC m_dims;
  void parse_npy_header(FILE* fp) {
    char buffer[256];
    int64_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11) {
      ERR << "Something wrong. Could not read header " << std::endl;
      exit(-1);
    }
    buffer[11] = NULL;
    std::cout << "first 11 characters::" << buffer << std::endl;

    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    // fortran order is column major order
    // C order is row major order
    int64_t loc1 = header.find("fortran_order");
    loc1 += 16;
    this->m_fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // obtain dimensions
    loc1 = header.find("(");
    int64_t loc2 = header.find(")");
    if (loc1 < 0 || loc2 < 0) {
      ERR << "could not find ()" << std::endl;
      exit(-1);
    }

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    if (str_shape[str_shape.size() - 1] == ',') {
      this->m_modes = 1;
    } else {
      this->m_modes = std::count(str_shape.begin(), str_shape.end(), ',') + 1;
    }
    this->m_dims = arma::zeros<UVEC>(m_modes);

    std::stringstream ss(str_shape);
    std::string s;
    ss.str(str_shape);
    int64_t i = 0;
    while (getline(ss, s, ',')) {
      this->m_dims[i++] = ::atoi(s.c_str());
    }
    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr");
    loc1 += 9;
    bool littleEndian =
        (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    std::string word_size = header.substr(loc1 + 2);
    loc2 = word_size.find("'");
    this->m_word_size = atoi(word_size.substr(0, loc2).c_str());
  }

 public:
  Tensor* m_input_tensor;
  NumPyArray() {
    this->m_word_size = 0;
    this->m_fortran_order = false;
    this->m_modes = 0;
  }
  void load(std::string fname) {
    FILE* fp = fopen(fname.c_str(), "rb");
    if (fp == NULL) {
      ERR << "Could not load the file " << fname << std::endl;
      exit(-1);
    }
    parse_npy_header(fp);
    this->m_input_tensor = new Tensor(this->m_dims);
    int64_t nread = fread(&m_input_tensor->m_data[0],
                          sizeof(std::vector<double>::value_type),
                          m_input_tensor->numel(), fp);
    if (nread != m_input_tensor->numel()) {
      WARN << "something wrong ::read::" << nread
           << "::numel::" << this->m_input_tensor->numel()
           << "::word_size::" << this->m_word_size << std::endl;
    }
  }
  void printInfo() {
    INFO << "modes::" << this->m_modes << "::dims::" << std::endl
         << this->m_dims << "::fortran_order::" << this->m_fortran_order
         << "::word_size::" << this->m_word_size << std::endl;
  }
};
}  // namespace planc

#endif  // COMMON_NPYIO_HPP_
