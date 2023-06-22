/* Copyright 2022 Ramakrishnan Kannan */
#ifndef JOINTNMFDRIVER_HPP_
#define JOINTNMFDRIVER_HPP_

#include "common/nmf.hpp"
#include <stdio.h>
#include <string>
#include "common/parsecommandline.hpp"
#include "common/utils.hpp"
#include "jointnmf/anlsbppjnmf.hpp"
#include "jointnmf/pgdjnmf.hpp"
#include "jointnmf/pgncgjnmf.hpp"

// TODO: move class defintion into here

    template <class NMFTYPE, class T1, class T2>
    void callJointNMF();

    // template parameters with template paramters
    template <template <class, class> class NMFTYPE>
    void build_opts();

#endif  // JOINTNMFDRIVER_HPP_