/Users:/ { users = $3 }
/Towers:/ { towers = $3 }
/recover traces/ { recover_time = $2 }
/map traces recovered/ { map_time = $2 }
/Overall accuracy:/ { accuracy = $3 }
/Overall accuracy std:/ { accuracy_std = $4 }
/Overall error:/ { error = $3 }
/Overall error std:/ { error_std = $4 }
/complete evaluation/ { print towers "\t" users "\t" accuracy "\t" accuracy_std "\t" error "\t" error_std "\t" recover_time "\t" map_time "\t" $2 }
