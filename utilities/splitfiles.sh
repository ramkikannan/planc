#!/bin/bash
#It splits the dense matrix into row splits and column splits
#Every file will have equal number of lines in Arows_#cores_idx
#and equal columsn in Acols_#cores_idx
#Run as : splitfiles.sh filename filenamet #cores
numCores=$3
totalLines=$(wc -l < $1)
#totalLines=
numLines=$(( $totalLines / $numCores ))
outputdir=${numCores}cores
rm -rf $outputdir
mkdir $outputdir
#------------------------------
#row split
split -a 3 -d -l ${numLines} $1 $outputdir/Arows_${numCores}_

#rename the file suffixes with numbers
i=0
for file in $outputdir/Arows_*
do
  mv $file $outputdir/Arows_${numCores}_${i}
  i=$(( i + 1 ))
done

totalLines=$(wc -l < $2)
#totalLines=115200
numLines=$(( $totalLines / $numCores ))
#------------------------------
#row split
split -a 3 -d -l ${numLines} $2 $outputdir/Acols_${numCores}_

#rename the file suffixes with numbers
i=0
for file in $outputdir/Acols_*
do
  mv $file $outputdir/Acols_${numCores}_${i}
  i=$(( i + 1 ))
done
#-------------------------------
#column split
#numCols=$(gawk -F' ' '{print NF; exit}' $1)
#perCol=$(( ${numCols} / ${numCores} ))
#for ((i=0;i<numCores;i++ ))
#do
  #j=$((i+1))
  #cut -f$((((${i}*${perCol})+1)))-$((((${i}+1)*${perCol}))) -d " " $1 >> ${outputdir}/Acols_${numCores}_$i 2>&1
#done
#-------------------------------------