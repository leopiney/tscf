BEGIN { print "towers,sampling,district_size,accuracy,accuracy_std,error,error_std,execution_time" }

/Users:/ { users = $3 }
/Towers:/ { towers = $3 }
/> Sampling:/ { sampling = $3 }
/> Accuracy:/ { district_size = $3 }
/Overall accuracy:/ { accuracy = $3 }
/Overall accuracy std:/ { accuracy_std = $4 }
/Overall error:/ { error = $3 }
/Overall error std:/ { error_std = $4 }

/complete evaluation with accuracy/ {
	execution_time = $2;
	print towers "," sampling "," district_size "," accuracy "," accuracy_std "," error "," error_std "," execution_time;
}
