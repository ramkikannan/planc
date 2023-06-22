# EXAMPLE: python3 run_partitioner.py configs/partition_configs.json example_partition

import subprocess as sb         #for running PLANC
import sys
from scipy.io import mmread
import json
import numpy as np
from scipy import sparse
from planc_partitioner import planc_partitioner

config_path = sys.argv[1]
f = open(config_path)
config = json.load(f)
f.close()

id = sys.argv[2]
c = config[id][0]

print("config type: " + str(type(config)))
print(list(config.keys()))

print(id)
print(c)

print(c['format'])

# partition feature matrix... ----> writing new partition function
test = planc_partitioner() 
output_path = c['partitioner_output_path'] if ('partitioner_output_path' in c) else c['feature_mat_path']

# automatically create output directory...
sb.run("mkdir " + output_path, shell=True)

# read in dataset ----> mm stands for matrix market format...
if c['format'] == "mm" :
    print("reading feature matrix in matrix market format")
    A = mmread(c['feature_mat_path'])
    m = A.shape[0]
    n = A.shape[1]
    print(type(A))
    print(A)

    test.partition(c['feature_mat_path'], output_path, A,m,n,c['pr'], c['pc'])
# elif c['format'] == "bin" :
#     print("reading feature matrix in binary format")
#     mat = np.fromfile(c['feature_mat_path'])
#     m = c['m']
#     n = c['n']

#     print(type(A))
#     print(A)

#     test.partition(output_path, A,m,n,c['pr'], c['pc'])
