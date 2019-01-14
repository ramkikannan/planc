#!/usr/bin/bash
SRC_DIR=$1
SYSTEM="$(hostname --long | \
sed -e 's/\.\(olcf\|ccs\)\..*//' \
-e 's/[-]\?\(login\|ext\|batch\)[^\.]*[\.]\?//' \
-e 's/[-0-9]*$//')"
echo $SYSTEM

#load modules RHEA
if [ "$SYSTEM" = "rhea" ];
then
    module unload PE-intel
    module load PE-gnu
    module swap gcc gcc/6.2.0
fi

#load modules EOS
if [ "$SYSTEM" = "eos" ];
then
    module unload PrgEnv-intel
    module load PrgEnv-gnu
    module swap gcc gcc/6.3.0
    module load cmake3
fi

#load modules EOS
if [ "$SYSTEM" = "titan" ];
then
    module unload PrgEnv-pgi
    module load PrgEnv-gnu
    module swap gcc gcc/6.3.0
    module load cmake3
    module load cudatoolkit
fi

for cfg in dense_nmf dense_ntf dense_distnmf dense_distntf sparse_nmf sparse_distnmf;
do    
    mkdir ../build_$SYSTEM\_$cfg
done
#dense builds
for cfg in nmf ntf distnmf distntf;
do
    echo $SYSTEM
    echo $cfg
    echo build_$SYSTEM\_dense_$cfg
    pushd ../build_$SYSTEM\_dense_$cfg
    if [ "$SYSTEM" = "rhea" ]; then
        cmake $SRC_DIR/$cfg/
    fi
    #we consider this as eos/titan    
    if [ "$SYSTEM" = "eos" ]; then
        CC=CC CXX=CC cmake $SRC_DIR/$cfg/ -DCMAKE_IGNORE_MKL=1
    fi

    if [ "$SYSTEM" = "titan" ]; then
        CC=CC CXX=CC cmake $SRC_DIR/$cfg/ -DCMAKE_IGNORE_MKL=1 -DCMAKE_BUILD_CUDA=1
    fi
    make
    popd
done
#sparse builds
for cfg in nmf distnmf;
do
    pushd ../build_$SYSTEM\_sparse_$cfg/
    if [ "$SYSTEM" = "rhea" ]; then
        cmake $SRC_DIR/$cfg/ -DCMAKE_BUILD_SPARSE=1
    fi
    #we consider this as eos/titan
    if [ "$SYSTEM" = "eos" ]; then
        CC=CC CXX=CC cmake $SRC_DIR/$cfg/ -DCMAKE_IGNORE_MKL=1 -DCMAKE_BUILD_SPARSE=1
    fi

    if [ "$SYSTEM" = "titan" ]; then
        CC=CC CXX=CC cmake $SRC_DIR/$cfg/ -DCMAKE_IGNORE_MKL=1 -DCMAKE_BUILD_SPARSE=1 -DCMAKE_BUILD_CUDA=0
    fi    
    make
    popd
done

#copy all the executable
cp ../build_$SYSTEM\_dense_nmf/nmf dense_nmf
cp ../build_$SYSTEM\_dense_ntf/ntf dense_ntf 
cp ../build_$SYSTEM\_dense_distnmf/distnmf dense_distnmf
cp ../build_$SYSTEM\_dense_distntf/distntf dense_distntf
cp ../build_$SYSTEM\_sparse_nmf/nmf sparse_nmf
cp ../build_$SYSTEM\_sparse_distnmf/distnmf sparse_distnmf
