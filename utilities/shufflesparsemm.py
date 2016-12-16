import numpy as np
import scipy.sparse as sp
import sys
import getopt

###############################################################################
# This file takes a sparse matrix as input file and shuffles it
# python shufflesparsemm.py -i inputfile -o outputfile
# There will be three outputfiles -- shuffled matrix, row and col permutation
###############################################################################

def save_sparse_matrix(filename,x):
    [rowidx,colidx,val]=sp.find(x)
    y = np.column_stack((rowidx,colidx,val))
    print 'saving shuffled matrix'
    np.savetxt(filename,y,fmt='%u',delimiter=' ')

def load_sparse_matrix(filename):
    y=np.loadtxt(filename)
    num_rows=max(y[:,0])+1
    num_cols=max(y[:,1])+1
    print 'loading input matrix'
    z=sp.csr_matrix((y[:,2], (y[:,0],y[:,1])),shape=(num_rows,num_cols), dtype=np.int32)
    return z

def randomize_matrix(input_file_name,output_file_name):
    z=load_sparse_matrix(input_file_name)
    print 'loaded input matrix'
    shape=z.shape
    rowperm=np.arange(0,shape[0],dtype=np.int32)
    np.random.shuffle(rowperm)
    colperm=np.arange(0,shape[1],dtype=np.int32)
    np.random.shuffle(colperm)
    z_rnd=z[rowperm,:]
    z_rnd=z_rnd[:,colperm]
    print 'shuffled input matrix'
    save_sparse_matrix(output_file_name, sp.csr_matrix.tocoo(z_rnd))
    np.savetxt(output_file_name+ '_rowperm', rowperm,fmt='%u')
    np.savetxt(output_file_name+'_colperm', colperm,fmt='%u')

def main(argv):
   inputfile=''
   outputfile=''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'shufflesparsemm -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
          print 'shufflesparsemm -i <inputfile> -o <outputfile>'
          sys.exit()
      elif opt in ("-i", "--ifile"):
          inputfile = arg
      elif opt in ("-o", "--ofile"):
          outputfile = arg
   randomize_matrix(inputfile,outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])