import math
from itertools import accumulate
import re
import os

class planc_partitioner:
    def __init__(self):
        pass

    def partition(self,feature_mat_path, output_path, A, nrows, ncols, pr, pc):
        pass

        print('hello from partition function')
        
        t = A.tocsr()

        n = self.itersplit(nrows,ncols,pr,pc)
        print(n)

        op = os.path.join(output_path, os.path.basename(feature_mat_path))

        for i in range(pr):
            for j in range(pc):
                pass
                print('proc: (' + str(i) + ',' + str(j) + ')')
                print('row start indices: ' + str(n[0][i]) + '  row end indices: ' + str(n[0][i+1]))
                print('col start indices: ' + str(n[1][j]) + '  col end indices: ' + str(n[1][j+1]))

                temp = t[n[0][i]:(n[0][i+1]), n[1][j]:(n[1][j+1])]

                # save matrix i_j ----> assuming that csr form is correct format...
                # https://stackoverflow.com/questions/55690069/how-to-write-a-sparse-matrix-in-a-text-file-in-python
                temp.maxprint = temp.count_nonzero() 
                
                with open(op + str(i) + '_' + str(j), 'w') as f:
                    mat = str(temp)
                    mat = mat.replace(')','').replace('(','').replace(',','')

                    f.write(mat)
                    f.close()

        return op


        


    def itersplit(self,nrows,ncols,pr,pc):
        pass

        n = [[0] * pr, [0] * pc] 
        print(n)

        row_div = math.floor(nrows / pr)
        col_div = math.floor(ncols / pc)

        row_mod = nrows % pr
        col_mod = ncols % pc

        for row in range(len(n[0])):
            pass
            # print('(' + str(row) + ',' + str(col) + ')')
            if row < row_mod:
                n[0][row] = row_div + 1
            else:
                n[0][row] = row_div

        for col in range(len(n[1])):
            pass
            if col < col_mod:
                n[1][col] = col_div + 1
            else:
                n[1][col] = col_div

        #prefix sum 
        n[0] = list(accumulate(n[0]))
        n[1] = list(accumulate(n[1]))

        n[0].insert(0,0)
        n[1].insert(0,0)

        return n