#!/bin/bash

for file in `ls $1/*_log.txt`; do
  niter=`cat $file|grep Valid|wc -l` 
  perf=`cat $file|grep Valid|tail -n 1`
  file=`basename $file`
  printf "%-100s --> (%s): %s" "$file" "$niter" "$perf"
  echo
done
