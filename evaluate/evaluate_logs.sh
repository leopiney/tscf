#
# Final tests for districts evaluation
#
array=( 0 1 2 3 4 5 6 7 8 9 ) 

for i in "${array[@]}"
do
  awk -f extract_logs.awk evaluate_districts_$i.log > evaluate_districts_$i.csv
done
