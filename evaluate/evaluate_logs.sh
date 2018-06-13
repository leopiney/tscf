awk -f extract_logs.awk evaluate_custom.log > evaluate_custom.csv
awk -f extract_logs.awk evaluate_random_direction.log > evaluate_random_direction.csv
awk -f extract_logs.awk evaluate_random_waypoint.log > evaluate_random_waypoint.csv

array=( 3 4 5 6 7 8 9 10 )

for i in "${array[@]}"
do
  awk -f extract_logs.awk evaluate_custom_$i.log > evaluate_custom_$i.csv
done

#
# Real tests evaluation for random direction and random waypoint
#

array=( 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39)

for i in "${array[@]}"
do
  awk -f extract_logs.awk evaluate_random_direction_$i.log > evaluate_random_direction_$i.csv
done

for i in "${array[@]}"
do
  awk -f extract_logs.awk evaluate_random_waypoint_$i.log > evaluate_random_waypoint_$i.csv
done