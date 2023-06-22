# OpenMP Joint Nonnegative Matrix Factorization 

## Install Instructions


This program depends on:

- [Armadillo](https://arma.sourceforge.net) library for linear algebra operations.
- [OpenBLAS](https://github.com/xianyi/OpenBLAS) for optimized kernels. If building with OpenBLAS and MKL is discoverable by `cmake`, use `-DCMAKE_IGNORE_MKL=1` while building. 

Once you have installed these libraries set the following environment variables.

````
export ARMADILLO_INCLUDE_DIR=/home/rnu/libraries/armadillo-6.600.5/include/
export LIB=$LIB:/home/rnu/libraries/openblas/lib:
export INCLUDE=$INCLUDE:/home/rnu/libraries/openblas/include:$ARMADILLO_INCLUDE_DIR:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rnu/libraries/openblas/lib/:
export CPATH=CPATH:$INCLUDE:
````

* Create a build directory. 
* Change to the build directory 
* In case of MKL, `source` the ````$MKLROOT/bin/mkl_vars.sh intel64````
* run `cmake` [PATH TO THE CMakeList.txt]
* `make`

Refer to the build scripts found in the [build](build/README.md) directory for a sample script.

### Input types
JointNMF requires two input matrices, the features and connections matrices, to run on.
PLANC is able to support any sparsity combination of the inputs to run on be default.
However, specialized builds can be constructed for each type if needed via the following `cmake` flags.
1. `-DCMAKE_BUILD_SPSP=1` sets the build to expect sparse features and sparse connections matrices.
2. `-DCMAKE_BUILD_SPDEN=1` sets the build to expect sparse features and dense connections matrices.
3. `-DCMAKE_BUILD_DENSP=1` sets the build to expect dense features and sparse connections matrices.
4. `-DCMAKE_BUILD_DENDEN=1` sets the build to expect dense features and dense connections matrices.

### Intel MKL vs OpenBLAS

- ````export LD_LIBRARY_PATH=MKL_LIB path````
- `source` the ````$MKLROOT/bin/mkl_vars.sh intel64````

## Runtime usage

Tell OpenBLAS how many threads you want to use. For example on a quad core system use the following.

```
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MKL_LIB
```

## Command Line options
The specialized command line options for the JointNMF binaries are given below.
* `--mat_type`: An array parameter specifying the sparsity type of the input matrices.
* `--alpha`: This parameter modifies the objective function to control the importance given to each of the fit terms (features and connections).
* `--beta`: The symmetric regularization parameter in the ANLS algorithm.
* `--gamma`: The momentum term for the PGD algorithm. 

## Citation

Please refer to the [papers](papers.md) section for the appropriate reports to cite.
