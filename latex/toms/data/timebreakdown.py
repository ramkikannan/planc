import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 4:
    print 'Usage: python timebreakdown.py input_file output_file title'
    sys.exit()

input_file=sys.argv[1]
output_file=sys.argv[2]
title=sys.argv[3]
print 'making plot for ' + input_file 

df1=pd.read_csv(input_file, delimiter=r"\s+")
df2 =df1[['alg-p','gram','nnls','mttkrp','multittv','reducescatter','allgather','allreduce']]
df2=df2.set_index('alg-p')

f = plt.figure()
plt.title(title, color='black')
df2.plot(kind='bar', stacked=True, ax=f.gca())
plt.legend(loc=9, ncol=3)
plt.ylabel('Running Time (in Secs)')
plt.gcf().set_size_inches(9, 8)
plt.savefig(output_file,format='pdf')

print 'check output file ' + output_file

