#!/bin/bash
#It splits the dense matrix into row splits and column splits
#Every file will have equal number of lines in Arows_#cores_idx
#and equal columsn in Acols_#cores_idx
#Run as : splitfiles.sh filename #pr #pc
pr=$2
pc=$3
totalLines=$(wc -l < $1)
numLines=$(( $totalLines / $pr ))
outputdir=${pr}cores
rm -rf $outputdir
mkdir $outputdir
#------------------------------
#row split
split -a 3 -l ${numLines} $1 $outputdir/Arows_${pr}_

#rename the file suffixes with numbers
i=0
for file in $outputdir/Arows_*
do
  mv $file $outputdir/Arows_${pr}_${i}
  i=$(( i + 1 ))
done
i=0
#-------------------------------
#column split
for file in $outputdir/Arows_*
do  
  numCols=$(gawk -F' ' '{print NF; exit}' $file)
  perCol=$(( ${numCols} / ${pc} ))
  num1=$(( ${i} * ${pc} ))
  i=$(( i + 1 ))
  for ((j=0;j<pc;j++ ))
  do
    #j=$((i+1))     
    idx=$(( num1 + j ))
    cut -f$((((${j}*${perCol})+1)))-$((((${j}+1)*${perCol}))) -d " " $file >> ${outputdir}/A_${idx} 2>&1
  done  
done

rm $outputdir/Arows_*
#-------------------------------------

