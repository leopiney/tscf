#!/usr/bin/env bash
array=( 0 1 2 3 4 5 6 7 8 9 ) 

for i in "${array[@]}"
do
  awk -f extract_logs.awk evaluations/evaluation_$i.log > evaluations/evaluation_$i.csv
done
