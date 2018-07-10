BEGIN { print "velocity,users,towers,sampling,district_size,accuracy,accuracy_std,error,error_std,execution_time" }

/Users:/ { users = $3 }
/Towers:/ { towers = $3 }
/Velocity:/ { velocity = "\""$3" "$4"\"" }
/> Sampling:/ { sampling = $3 }
/> Accuracy:/ { district_size = $3 }
/NEW Overall accuracy:/ { accuracy = $4 }
/NEW Overall accuracy std:/ { accuracy_std = $5 }
/NEW Overall error:/ { error = $4 }
/NEW Overall error std:/ { error_std = $5 }

/recover traces from aggregated data/ { recover_time = $2 }
/complete evaluation with accuracy/ {
	mapping_time = $2
	execution_time = mapping_time + recover_time;
	print velocity "," users "," towers "," sampling "," district_size "," accuracy "," accuracy_std "," error "," error_std "," execution_time;
}
