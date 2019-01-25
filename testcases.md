# Testcases

Sample test cases to execute are. You should see the error going down. We also tested the scaling up on a relatively small size of matrix say 20000 by 10000 of dense. The scale out on large number of cluster with big size matrix also works good. Ensure that the objective error is decreasing at every iteration. 

Openmp Dense NMF:
=================
1. ./dense_nmf -d "2000 2000" -t 10 -k 10 -a 0 -e 1
2. ./dense_nmf -d "2000 2000" -t 10 -k 10 -a 1 -e 1
3. ./dense_nmf -d "2000 2000" -t 10 -k 10 -a 2 -e 1

Openmp Dense NTF:
=================
1. ./dense_ntf -d "200 200 200 200" -t 10 -k 10 -a 2 -e 1

Openmp Dense NMF:
=================
1. ./sparse_nmf -d "2000 2000" -t 10 -k 10 -a 0 -e 1 -s 0.01
2. ./sparse_nmf -d "2000 2000" -t 10 -k 10 -a 1 -e 1 -s 0.01
3. ./sparse_nmf -d "2000 2000" -t 10 -k 10 -a 2 -e 1 -s 0.01

Dist Sparse NMF:
================
1. mpirun -np 16 ./sparse_distnmf -d "2000 2000" -k 50 -t 10 -i rand_lowrank  -p "4 4"  -a 0 -e 1 -s 0.01
2. mpirun -np 16 ./sparse_distnmf -d "2000 2000" -k 50 -t 10 -i rand_lowrank  -p "4 4"  -a 1 -e 1 -s 0.01
3. mpirun -np 16 ./sparse_distnmf -d "2000 2000" -k 50 -t 10 -i rand_lowrank  -p "4 4"  -a 2 -e 1 -s 0.01

Dist Dense NMF:
===============
1. mpirun -np 16 ./dense_distnmf -d "2000 2000" -k 50 -t 10 -i rand_lowrank  -p "4 4"  -a 0 -e 1
2. mpirun -np 16 ./dense_distnmf -d "2000 2000" -k 50 -t 10 -i rand_lowrank  -p "4 4"  -a 1 -e 1 
3. mpirun -np 16 ./dense_distnmf -d "2000 2000" -k 50 -t 10 -i rand_lowrank  -p "4 4"  -a 2 -e 1 

Dist Dense NTF:
===============
1. mpirun -np 16 ./dense_distntf -d "200 200 200 200" -k 50 -t 10 -i rand_lowrank  -p "2 2 2 2" -a 2 -e 1
