array=( 30 31 32 33 34 35 36 37 38 39 )

for i in "${array[@]}"
do
  python -m tracer.evaluate run random_direction > evaluate/evaluate_random_direction_$i.log &
done

for i in "${array[@]}"
do
  python -m tracer.evaluate run random_waypoint > evaluate/evaluate_random_waypoint_$i.log &
done