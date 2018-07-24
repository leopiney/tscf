#!/usr/bin/env bash
array=( 0 1 2 3 4 5 6 7 8 9)

mkdir -p evaluations

for i in "${array[@]}"
do
  python -u -m tracer.evaluate run > evaluations/evaluation_$i.log &
done
