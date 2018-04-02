#!/usr/bin/python2.7
# Read a matrix in coordinate format and create another matrix file with sparse struct headers

import sys

matname = sys.argv[1]
outmatname = sys.argv[2]


print matname

f = open(matname, "r")
nz = 0;
nrows = 0;
ncols = 0;
for line in f:
  line = line.split()
  nz = nz + 1
  line[0] = int(line[0])
  line[1] = int(line[1])
  if line[0] + 1 > nrows:
    nrows = line[0] + 1
  if line[1] + 1 > ncols:
    ncols = line[1] + 1
  if nz % 100000 == 0:
    print nz
f.close()

f = open(matname, "r")
fo = open(outmatname, "w")
fo.write('%d %d\n' % (2, nz))
fo.write('%d %d\n' % (nrows, ncols))
nz = 0;
for line in f:
  line = line.split()
  line[0] = int(line[0]) + 1
  line[1] = int(line[1]) + 1
  if len(line) == 2:
    val = 1.0
  else:
    val = float(line[2]);
  nz = nz + 1
  if nz % 100000 == 0:
    print nz
  fo.write('%d %d %lf\n' % (line[0], line[1], val))

fo.close()
f.close()
