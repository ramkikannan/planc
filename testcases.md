Sample test cases to execute are. You should see the error going down. We also tested the scaling up on a relatively small size of matrix say 20000 by 10000 of dense. The scale out on large number of cluster with big size matrix also works good. 

Openmp:
======
1. ./nmf -m 2000 -n 2000 -t 10 -k 10 -a 0
2. ./nmf -m 2000 -n 2000 -t 10 -k 10 -a 1
3. ./nmf -m 2000 -n 2000 -t 10 -k 10 -a 2

MPI Sparse:
==========
1. mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 0 -s 0.01
2. mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 1 -s 0.01
3. mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 2 -s 0.01

MPI Dense:
==========
1. mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 0
2. mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 1 
3. mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 2 


