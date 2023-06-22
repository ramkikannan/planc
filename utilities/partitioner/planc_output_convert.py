# Copyright Benjamin Cobb 2022
# NOTE: the goal of this script is to convert planc output to either .txt, .mtx, or .npz formats
import numpy as np
import scipy
from scipy.io import mmwrite
import sys
import os

# USAGE: python3 planc_output_convert.py path_to_planc_output converted_output_path conversion_format m n k
# python3 planc_output_convert.py ./test_data/output/11_9_test_10_2_2 ./test_data/converted/ npz 11 9 2

try:
    planc_output_path = sys.argv[1]
    converted_output_path = sys.argv[2]
    conversion_format = sys.argv[3]
    m = int(sys.argv[4])
    n = int(sys.argv[5])
    k = int(sys.argv[6])
except:
    print('USAGE:   python3 planc_output_convert.py path_to_planc_output converted_output_path conversion_format m n k')
    print('EXAMPLE: python3 planc_output_convert.py ./test_data/output/11_9_test_10_2_2 ./test_data/converted/ npz 11 9 2')
    exit()

W_path = planc_output_path + '_W'
H_path = planc_output_path + '_H'
print(W_path)
print(H_path)

prefix = converted_output_path + os.path.basename(planc_output_path)
W_converted = prefix + '_W'
H_converted = prefix + '_H'
print('prefix: ' + prefix)
print(W_converted)
print(H_converted)

# NOTE: is float64 the correct data-type?...
W = np.fromfile(W_path, 'float64')
W = np.asmatrix(W).reshape(k, m)


H = np.fromfile(H_path, 'float64')
H = np.asmatrix(H).reshape(k,n)

if conversion_format == "mtx":
    pass
    mmwrite(W_converted, W)
    mmwrite(H_converted, H)
elif conversion_format == "txt":
    pass
    np.savetxt(W_converted + '.txt', W)
    np.savetxt(H_converted + '.txt', H)
elif conversion_format == "npz":
    pass
    np.savez(W_converted, W)
    np.savez(H_converted, H)
else:
    print('conversion format not recognized, options are: mtx, txt, and npz')