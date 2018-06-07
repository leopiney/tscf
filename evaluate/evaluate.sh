array=( 20 21 22 23 24 25 26 27 28 29 )

for i in "${array[@]}"
do
  python -m tracer.evaluate run random_direction > evaluate/evaluate_random_direction_$i.log &
done

for i in "${array[@]}"
do
  python -m tracer.evaluate run random_waypoint > evaluate/evaluate_random_waypoint_$i.log &
done