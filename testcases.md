Sample test cases to execute are. You should see the error going down. We also tested the scaling up on a relatively small size of matrix say 20000 by 10000 of dense. The scale out on large number of cluster with big size matrix also works good. 

Openmp:
=======
* OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 ./nmf -m 50000 -n 20000 -t 10 -k 10 -a 0
* OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 ./nmf -m 50000 -n 20000 -t 10 -k 10 -a 1
* OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 ./nmf -m 50000 -n 20000 -t 10 -k 10 -a 2
 
MPI Sparse:
===========
* mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 0 -s 0.01
* mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 1 -s 0.01
* mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 2 -s 0.01
 
MPI Dense:
==========
* mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 0
* mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 1
* mpirun -np 16 ./distnmf -m 2000 -n 2000 -k 50 -t 10 -i rand_lowrank  --pr=4 --pc=4 -e 1 -a 2
 

