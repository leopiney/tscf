BEGIN { print "towers,sampling,district_size,accuracy,accuracy_std,error,error_std,execution_time" }

/Users:/ { users = $3 }
/Towers:/ { towers = $3 }
/> Sampling:/ { sampling = $3 }
/> Accuracy:/ { district_size = $3 }
/Overall accuracy:/ { accuracy = $3 }
/Overall accuracy std:/ { accuracy_std = $4 }
/Overall error:/ { error = $3 }
/Overall error std:/ { error_std = $4 }

/recover traces from aggregated data/ { recover_time = $2 }
/complete evaluation with accuracy/ {
	mapping_time = $2
	execution_time = mapping_time + recover_time;
	print towers "," sampling "," district_size "," accuracy "," accuracy_std "," error "," error_std "," execution_time;
}
