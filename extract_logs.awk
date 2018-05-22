BEGIN { print "velocity,towers,users,accuracy,accuracy_std,error,error_std,recover_time,map_time,execution_time" }

/Users:/ { users = $3 }
/Towers:/ { towers = $3 }
/Sigma:/ { velocity = $3 }
/Velocity:/ { velocity = "\""$3" "$4"\"" }
/recover traces/ { recover_time = $2 }
/map traces recovered/ { map_time = $2 }
/Overall accuracy:/ { accuracy = $3 }
/Overall accuracy std:/ { accuracy_std = $4 }
/Overall error:/ { error = $3 }
/Overall error std:/ { error_std = $4 }

/complete evaluation/ {
	execution_time = $2;
	print velocity "," towers "," users "," accuracy "," accuracy_std "," error "," error_std "," recover_time "," map_time "," execution_time;
}
