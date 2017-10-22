#!/bin/bash

if [ "$#" -eq 2 ]; then
  niter=$2
else
  niter=100
fi

for file in `ls $1/*txt`; do
  niter=`cat $file|grep Valid|head -n $niter|wc -l` 
  perf=`cat $file|grep Valid|head -n $niter|tail -n 1`
  file=`basename $file`
  printf "%-100s --> (%s): %s" "$file" "$niter" "$perf"
  echo
done
