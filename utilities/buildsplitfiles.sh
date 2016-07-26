g++ -fopenmp -c -O3 -I /home/rkannan3/libraries/armadillo-4.450.3/include/ -I /home/rkannan3/Documents/gatech/NMFLibrary/ -L /home/rkannan3/libraries/armadillo-4.450.3/usr/lib/ -larmadillo ~/NMFLibrary/matrix_market_file.cpp
g++ -fopenmp -c -O3 -I /home/rkannan3/libraries/armadillo-4.450.3/include/ -I /home/rkannan3/NMFLibrary/ -L /home/rkannan3/libraries/armadillo-4.450.3/usr/lib/ -larmadillo SplitFiles.cpp
g++ -fopenmp -O3 SplitFiles.o matrix_market_file.o -o SplitFiles
