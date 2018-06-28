#
# Final tests for districts evaluation
#
array=( 0 1 2 3 ) 

for i in "${array[@]}"
do
  awk -f extract_logs.awk evaluate_districts_$i.log > evaluate_districts_$i.csv
done
