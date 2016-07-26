export MKL_ROOT=/opt/intel/composer_xe_2015.2.164/mkl/
source /opt/intel/composer_xe_2015.2.164/mkl/bin/mklvars.sh intel64
export MPI_C_LIBRARIES="-Wl,--start-group /opt/cray/mpt/default/gni/mpich2-gnu/5.1/lib/libmpich_gnu_51.a /opt/cray/pmi/default/lib64/libpmi.a /opt/cray/alps/default/lib64/libalpsutil.a /opt/cray/alps/default/lib64/libalpslli.a /opt/cray/alps/default/lib64/libalps.a /opt/cray/wlm_detect/default/lib64/libwlm_detect.a /opt/cray/ugni/default/lib64/libugni.a /opt/cray/udreg/default/lib64/libudreg.a /opt/cray/xpmem/default/lib64/libxpmem.a -lpthread /usr/lib64/librt.a -Wl,--end-group"
cmake $1 -DCMAKE_BUILD_SPARSE=1 -DMPI_CXX_INCLUDE_PATH=/opt/cray/mpt/default/gni/mpich2-gnu/5.1/include -DMPI_CXX_LIBRARIES="$MPI_C_LIBRARIES"

