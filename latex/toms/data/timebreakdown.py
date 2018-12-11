import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import colors as mcolors

if len(sys.argv) < 4:
    print 'Usage: python timebreakdown.py input_file output_file title'
    sys.exit()

input_file=sys.argv[1]
output_file=sys.argv[2]
title=sys.argv[3]
print 'making plot for ' + input_file 

df1=pd.read_csv(input_file, delimiter=r"\s+", comment='#')
df2 =df1[['alg-p','gram','nnls','mttkrp','multittv','reducescatter','allgather','allreduce']]
df2=df2.set_index('alg-p')

f = plt.figure()
plt.title(title, color='black')
#my_colors = 'rgbymcw'
#ax = df2.plot(kind='bar', stacked=True, ax=f.gca(), color=my_colors)
ax = df2.plot(kind='bar', stacked=True, ax=f.gca())
bars = ax.patches
patterns =(None, None, None, None, '-', '+', 'x') #,'/','//','O','o','\\','\\\\')
#patterns_color = (None, None, None, None, 'r', 'g', 'b')
#color_types=('r', 'g', 'b', 'm', None, None, None)
hatches = [p for p in patterns for i in range(len(df2))]
#hatch_colors = [p for p in patterns_color for i in range(len(df2))]
#fill_colors = [p for p in color_types for i in range(len(df2))]
#for bar, hatch, hatch_color, fill_color in zip(bars, hatches): # hatch_colors, fill_colors):
for bar, hatch in zip(bars, hatches): # hatch_colors, fill_colors):
    bar.set_hatch(hatch)
    #bar.set_color(fill_color)
    #bar.set_hatch_color(mcolors.rgb_to_hsv(hatch_color))
plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left', ncol=1)
plt.ylabel('Running Time (in Secs)')
#plt.gcf().set_size_inches(9, 8)
plt.savefig(output_file,format='pdf',bbox_inches='tight')

print 'check output file ' + output_file

