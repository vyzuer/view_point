#!/bin/csh -f
# dataset path, weather data path

if($#argv != 2 ) then
    echo "pass dataset path"
    exit()
endif

set dbPath = $argv[1]
set weatherData = $argv[2]

pushd $dbPath  # [[ dbpath

set image_details = "images.details"
set file_name = "weather.info"
set lines = `cat $image_details`

unlink $file_name

set i = 13

while($i < $#lines)
#while($i < 38)

    set ix = `echo $i | awk '{print $1+11}'`
    set d = `echo $i | awk '{print $1+8}'`
    set t = `echo $i | awk '{print $1+9}'`

    set date = $lines[$d]
    set mytime = $lines[$t]

    set year = `echo $date | cut -d'-' -f 1 -s`
    set month = `echo $date | cut -d'-' -f 2 -s | awk '{print 1*$1}'`
    set day = `echo $date | cut -d'-' -f 3 -s | awk '{print 1*$1}'`

    set hour = `echo $mytime | cut -d':' -f 1 | awk '{print 1*$1}'`
    set mins = `echo $mytime | cut -d':' -f 2 | awk '{print 1*$1}'`
    
    set min_s = `echo $mins | awk '{print int($1/15)}'`
    if($min_s == 2) then
        set min_s = 1
    else if($min_s == 3) then
        set min_s = 2
    endif
    
    set line_num = `echo $hour $min_s | awk '{print int(3 + 2*$1 + $2 + ($1-2)/3 + 1)}'`
    echo $line_num
    echo $year $month $day $hour $mins #>> $file_name

    set weather_db = "${weatherData}/WDB2/${year}/${month}/${day}.info"
    echo $weather_db


    set total_lines = `wc -l $weather_db | cut -d' ' -f 1`
    if($line_num > $total_lines) then
        set line_num = 3
    endif

    set line_data = `sed ''"$line_num"'q;d' $weather_db`
    echo $line_data

    set temp = `echo $line_data | cut -d',' -f 2`
    set dew = `echo $line_data | cut -d',' -f 3`
    set humidity = `echo $line_data | cut -d',' -f 4`
    set visibility = `echo $line_data | cut -d',' -f 6`
    set wind = `echo $line_data | cut -d',' -f 8`
    set events = `echo $line_data | cut -d',' -f 11`
    set conditions = `echo $line_data | cut -d',' -f 12`

    set temp_var = `echo $visibility | awk '{print int($1)}'`

    if("Calm" == $wind) then
        set wind = 0
    endif

    if( $temp_var < 0 ) then
        set visibility = 10
    endif

    set thunder = "0 0 0"
    set rain = "0 0 0"
    set cloud = "0 0 0 0"
    set haze = 0
    set mist = 0
    
    set temp_var = `echo $conditions | grep "Haze"`
    if( "xxx" != "xxx$temp_var") then
        set haze = 1
    endif

    set temp_var = `echo $conditions | grep "Mist"`
    if( "xxx" != "xxx$temp_var") then
        set mist = 1
    endif

    set temp_var = `echo $conditions | grep "Scattered Clouds"`
    if( "xxx" != "xxx$temp_var") then
        set cloud = "0 0 0 1"
    endif

    set temp_var = `echo $conditions | grep "Partly Cloudy"`
    if( "xxx" != "xxx$temp_var") then
        set cloud = "0 0 1 0"
    endif

    set temp_var = `echo $conditions | grep "Mostly Clouds"`
    if( "xxx" != "xxx$temp_var") then
        set cloud = "0 1 0 0"
    endif

    set temp_var = `echo $conditions | grep "Overcast"`
    if( "xxx" != "xxx$temp_var") then
        set cloud = "1 0 0 0"
    endif

    set temp_var = `echo $conditions | grep "Rain"`
    if( "xxx" != "xxx$temp_var") then
        set rain = "0 1 0"
    endif

    set temp_var = `echo $conditions | grep "Light Rain"`
    if( "xxx" != "xxx$temp_var") then
        set rain = "0 0 1"
    endif

    set temp_var = `echo $conditions | grep "Heavy Rain"`
    if( "xxx" != "xxx$temp_var") then
        set rain = "1 0 0"
    endif

    set temp_var = `echo $conditions | grep "Thunderstorm"`
    if( "xxx" != "xxx$temp_var") then
        set thunder = "0 1 0"
    endif

    set temp_var = `echo $conditions | grep "Light Thunderstorm"`
    if( "xxx" != "xxx$temp_var") then
        set thunder = "0 0 1"
    endif

    set temp_var = `echo $conditions | grep "Heavy Thunderstorm"`
    if( "xxx" != "xxx$temp_var") then
        set thunder = "1 0 0"
    endif

    set sun_data = "${weatherData}/SunMoonDB/${year}/${month}/sun_1.info"
    echo $sun_data

    set line_num = `echo $day | awk '{print 108 + $1}'`
    echo $line_num

    set line_data = `sed ''"$line_num"'q;d' $sun_data`
    echo $line_data

    set sunrise = $line_data[4]
    set sunrise_h = `echo $sunrise | cut -d':' -f 1 | awk '{print 1*$1}'`
    set sunrise_m = `echo $sunrise | cut -d':' -f 2 | awk '{print 1*$1}'`
    set sunrise = `echo $sunrise_h $sunrise_m | awk '{print $1 + $2/60.0}'`
#echo $sunrise

    set sunset = $line_data[5]
    set sunset_h = `echo $sunset | cut -d':' -f 1 | awk '{print 1*$1}'`
    set sunset_m = `echo $sunset | cut -d':' -f 2 | awk '{print 1*$1}'`
    set sunset = `echo $sunset_h $sunset_m | awk '{print $1 + $2/60.0}'`

    set sunpeak = $line_data[11]
    set sunpeak_h = `echo $sunpeak | cut -d':' -f 1 | awk '{print 1*$1}'`
    set sunpeak_m = `echo $sunpeak | cut -d':' -f 2 | awk '{print 1*$1}'`
    set sunpeak = `echo $sunpeak_h $sunpeak_m | awk '{print $1 + $2/60.0}'`


    set img_time = `echo $hour $mins | awk '{print $1 + $2/60.0}'`

    set time1 = `echo $img_time $sunrise | awk '{ print ((($1 - $2) >= 0) ? $1-$2 : $2-$1)}'`
    set time1 = `echo $time1 | awk '{print ($1 < 12) ? $1 : 24 - $1}'`

    set time2 = `echo $img_time $sunset | awk '{ print ((($1 - $2) >= 0) ? $1-$2 : $2-$1)}'`
    set time2 = `echo $time2 | awk '{print ($1 < 12) ? $1 : 24 - $1}'`

    set time3 = `echo $img_time $sunpeak | awk '{ print ((($1 - $2) >= 0) ? $1-$2 : $2-$1)}'`
    set time3 = `echo $time3 | awk '{print ($1 < 12) ? $1 : 24 - $1}'`

# echo "${temp}:${dew}:${humidity}:${visibility}:${wind}:${thunder}:${rain}:${cloud}:${haze}:${mist}:${time1}:${time2}:${time3}" >> $file_name
    
# echo "${temp} ${dew} ${humidity} ${visibility} ${wind} ${thunder} ${rain} ${cloud} ${haze} ${mist} ${time1} ${time2} ${time3}" >> $file_name
    echo "${temp} ${dew} ${humidity} ${visibility} ${thunder} ${rain} ${cloud} ${haze} ${mist} ${time1} ${time2} ${time3}" >> $file_name

    @ i = ($ix + 1)

end

popd # ]] dbpath
