# Testcases

Sample test cases to execute are. You should see the error going down. We also tested the scaling up on a relatively small size of matrix say 20000 by 10000 of dense. The scale out on large number of cluster with big size matrix also works good. Ensure that the objective error is decreasing at every iteration. 

Openmp Dense NMF:
=================
```
for a in 0 1 2 4 
do
./dense_nmf -d "2000 2000" -t 10 -k 10 -a $a -e 1 -i rand_lowrank
done
./dense_nmf -d "100 100" -k 10 -t 10 -e 1 -a 2 --symm 0.0 -i rand_lowrank
./dense_nmf -d "100 100" -k 10 -t 10 -e 1 -a 7 --symm 0.0 -i rand_lowrank
```

Openmp Dense NTF:
=================
```
for a in 0 1 2 4 5
do
./dense_ntf -d "200 200 200" -t 10 -k 10 -a $a -e 1 -i rand_lowrank
done
```

Openmp Sparse NMF:
=================
```
for a in 0 1 2 4 
do
./sparse_nmf -d "2000 2000" -t 10 -k 10 -a $a -e 1 -i rand_lowrank -s 0.01
done
./sparse_nmf -d "100 100" -k 10 -t 10 -e 1 -a 2 --symm 0.0 -i rand_lowrank -s 0.05
./sparse_nmf -d "100 100" -k 10 -t 10 -e 1 -a 7 --symm 0.0 -i rand_lowrank -s 0.05
```
Dist Dense NMF:
================
```
for a in 0 1 2 4 
do
mpirun -np 16 ./dense_distnmf -d "2000 2000" -p "4 4" -t 10 -k 10 -a $a -e 1 -i rand_lowrank
mpirun -np 16 ./dense_distnmf -d "2000 2000" -p "8 2" -t 10 -k 10 -a $a -e 1 -i rand_lowrank
mpirun -np 16 ./dense_distnmf -d "1500 1000" -p "4 4" -t 10 -k 10 -a $a -e 1 -i rand_lowrank
mpirun -np 16 ./dense_distnmf -d "1500 1000" -p "8 2" -t 10 -k 10 -a $a -e 1 -i rand_lowrank
done
mpirun -np 16 ./dense_distnmf -d "2000 2000" -p "4 4" -t 10 -k 10 -a 2 -e 1 -i rand_lowrank --symm 0.0
mpirun -np 16 ./dense_distnmf -d "2000 2000" -p "4 4" -t 10 -k 10 -a 7 -e 1 -i rand_lowrank --symm 0.0
mpirun -np 16 ./dense_distnmf -d "830 830" -p "4 4" -t 10 -k 10 -a 2 -e 1 -i rand_lowrank --symm 0.0
mpirun -np 16 ./dense_distnmf -d "830 830" -p "4 4" -t 10 -k 10 -a 7 -e 1 -i rand_lowrank --symm 0.0
```


Dist Sparse NMF:
===============
```
for a in 0 1 2 4 
do
mpirun -np 16 ./sparse_distnmf -d "2000 2000" -p "4 4" -t 10 -k 10 -a $a -e 1 -i rand_lowrank -s 0.01
mpirun -np 16 ./sparse_distnmf -d "2000 2000" -p "8 2" -t 10 -k 10 -a $a -e 1 -i rand_lowrank -s 0.01
mpirun -np 16 ./sparse_distnmf -d "1500 1000" -p "4 4" -t 10 -k 10 -a $a -e 1 -i rand_lowrank -s 0.01
mpirun -np 16 ./sparse_distnmf -d "1500 1000" -p "8 2" -t 10 -k 10 -a $a -e 1 -i rand_lowrank -s 0.01
done
mpirun -np 16 ./sparse_distnmf -d "2000 2000" -p "4 4" -t 10 -k 10 -a 2 -e 1 -i rand_lowrank -s 0.01 --symm 0.0
mpirun -np 16 ./sparse_distnmf -d "2000 2000" -p "4 4" -t 10 -k 10 -a 7 -e 1 -i rand_lowrank -s 0.01 --symm 0.0
mpirun -np 16 ./sparse_distnmf -d "830 830" -p "4 4" -t 10 -k 10 -a 2 -e 1 -i rand_lowrank -s 0.01 --symm 0.0
mpirun -np 16 ./sparse_distnmf -d "830 830" -p "4 4" -t 10 -k 10 -a 7 -e 1 -i rand_lowrank -s 0.01 --symm 0.0
```

Dist Dense NTF:
===============
```
for a in 0 1 2 4 5
do
mpirun -np 9 ./dense_ntf -d "200 200 200" -p "3 3 3" -t 10 -k 10 -a $a -e 1 -i rand_lowrank --dimtree 1 
done
```