/* Copyright 2022 Ramakrishnan Kannan */
#ifndef DISTJNMFDRIVER_HPP_
#define DISTJNMFDRIVER_HPP_

#include <string>
#include "common/distutils.hpp"
#include "common/parsecommandline.hpp"
#include "common/utils.hpp"
#include "distjointnmf/distjointnmfio.hpp"
#include "distjointnmf/distanlsbppjnmf.hpp"
#include "distjointnmf/jnmfmpicomm.hpp"
#include "distjointnmf/distpgdjnmf.hpp"
#include "distjointnmf/distpgncgjnmf.hpp"

// TODO: move classs defintion into here

    template <class NMFTYPE, class T1, class T2>
    void callDistJointNMF();

    // template parameters with template paramters
    template <template <class, class> class NMFTYPE>
    void build_opts();

#endif  // DISTJNMFDRIVER_HPP_