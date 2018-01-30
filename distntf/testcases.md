Sample test cases executed and verified for output.

* mpirun -np 1 ./distntf -a 2 -k 3 -d "8 8 8" -p "1 1 1" -i rand_lowrank -t 3 -e 1
* mpirun -np 8 ./distntf -a 2 -k 3 -d "64 64 64" -p "2 2 2" -i rand_uniform -t 30 -e 1
