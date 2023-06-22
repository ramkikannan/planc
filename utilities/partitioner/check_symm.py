# NOTE: this checks whether the output from rand_symm functions is actually symmetric...
import numpy as np
import sys
from planc_partitioner import planc_partitioner
from pandas import *
# import pyarma
# from pyarma import *

# USAGE: python3 check_symm.py path_to_rnd_output/prefix m n k pr pc print? dense?
# EXAMPLE: python3 check_symm.py ~/Desktop/Projects/PLANC/nmflibrary/build/build_joint/Srnd 23 11 2
# 3 2 true true
try:
    planc_output_path = sys.argv[1]
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    k = int(sys.argv[4])
    pr = int(sys.argv[5])
    pc = int(sys.argv[6])
    print_opt = sys.argv[7]
    dense = sys.argv[8]
except:
    print('USAGE: python3 check_symm.py path_to_rnd_output/prefix m n k pr pc print? dense?')
    exit()

print("dense = {}, print = {}".format(dense, print_opt))

A = np.zeros((n, n))
test = planc_partitioner()

b = test.itersplit(n, n, pr, pc)
print(b)

for i in range(pr):
    pass
    for j in range(pc):
        pass
    # NOTE: PLANC writes out random output in linear process order, not with the grid dimensions.
    # Here we assume that the linear indexing is in row-major order...
    # t1 = b[0][i] : (b[0][i+1])
    # t2 = b[1][j] : (b[1][j+1])
        # print(str(i) + ", " + str(j))
        file_path = planc_output_path + "_" + str(pr * pc) + "_" + str(i*pc + j)
        print(file_path)
        # A[b[0][i] : (b[0][i+1]), b[1][j] : (b[1][j+1])]    

        #  NOTE rand output is written out to file via armadillo, not mpi...
        # G = np.fromfile(file_path, 'float64')

        if dense == "true":
            #G = mat()
            #G.load(file_path)
            G = np.fromfile(file_path, dtype=np.float64, sep=" ")
            G.resize( b[0][i+1] - b[0][i], b[1][j+1] - b[1][j])
            #print(G)
            A[b[0][i] : (b[0][i+1]), b[1][j] : (b[1][j+1])] = G
        else:
            with open(file_path) as f:
                lines = f.readlines()
            # print(lines)
            for l in lines:
                t = l.split()
                A[b[0][i] + int(t[0])][b[1][j] + int(t[1])] = float(t[2])


np.set_printoptions(precision=2)
if print_opt == "true":
    print(DataFrame(A))
for i in range(n):
    for j in range(n):
        if A[i][j] != A[j][i]:
            raise Exception("(A[" + str(i) + "][" + str(j) + "] = " + str(A[i][j]) + ")   !=   (A[" + str(j) + "][" + str(i) + "] = " + str(A[j][i]) + ")" )

print("matrix is symmetric")
