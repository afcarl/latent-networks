#!/bin/bash
for file in `ls $1/*_log.txt`; do
  perf=`cat $file|grep Valid|tail -n 1`
  file=`basename $file`
  echo "$file --> $perf"
done
