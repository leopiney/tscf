array=( 0 1 2 3 4 5 6 7 8 9)

for i in "${array[@]}"
do
  python -u -m tracer.evaluate run > evaluate/evaluate_districts_$i.log &
done
