#!/bin/csh -f
# dataset path, weather data path

# ./genWeatherData.csh ~/Project/Flickr-YsR/merlionImages/ ~/Copy/Flickr-code/weatherDB/ /home/yogesh/Copy/Flickr-code/PhotographyAssistance/cameraParameters/DB/

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/
pushd $db_path

# set loc = $argv[1]

foreach location (forbiddencity tiananmen)
# foreach location (arcde colognecathedral gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican)
# foreach location (liberty merlion taj vatican)

    pushd $location
    echo $location

    set image_details = "images.details"
    set file_name = "weather.info"
    set lines = `cat $image_details`

    unlink $file_name
    
    set i = 14
    
    while($i < $#lines)
    #while($i < 38)
    
#         echo $location
        echo $lines[$i]
        set ix = `echo $i | awk '{print $1+12}'`
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
#         echo $min_s
        if($min_s == 2) then
            set min_s = 1
        else if($min_s == 3) then
            set min_s = 2
        else
            set min_s = 0
        endif
        
        set search_string = $hour
        set ampm = "AM"
        if( $hour > 12 ) then
            set search_string = `echo $hour | awk '{print $1 - 12}'`
            set ampm = "PM"
        else if($hour == 0) then
            set search_string = 12
        endif

        set back_up_string = "${search_string}:[0-9][0-9] ${ampm}"

        if ($min_s == 0) then
            set search_string = "${search_string}:[0-2][0-9] ${ampm}"
        else if($min_s == 1) then
            set search_string = "${search_string}:[3-5][0-9] ${ampm}"
        else if($min_s == 2 && $hour != 23) then
            if($search_string == 11) then
                set ampm = "PM"
            endif
            if($search_string == 12) then
                set search_string = 1
            else
                set search_string = `echo $search_string | awk '{print $1 + 1}'`
            endif
            set search_string = "${search_string}:[0-2][0-9] ${ampm}"
        else 
            set search_string = "${search_string}:[0-5][0-9] ${ampm}"
        endif

#         echo $search_string

        # set line_num = `echo $hour $min_s | awk '{print int(3 + 2*$1 + $2 + ($1-2)/3 + 1)}'`
        set weather_db = "WDB2/${year}/${month}/${day}.info"
        echo $weather_db
    
        set line_num = `grep -n -m 1 "$search_string" $weather_db | cut -d':' -f 1`
        set total_lines = `wc -l $weather_db | cut -d' ' -f 1`
        if ("xx$line_num" == "xx") then
            set line_num = `grep -n -m 1 "$back_up_string" $weather_db | cut -d':' -f 1`
            if("xx$line_num" == "xx") then
                echo $location $weather_db $min_s $hour $mins $lines[$i]
                set line_num = `echo $hour | awk '{print 3 + $1}'`
            endif
        endif
#         echo $line_num
#         echo $year $month $day $hour $mins #>> $file_name
    
        set line_data = `sed ''"$line_num"'q;d' $weather_db`
#        echo $line_data

        set rain_line = `echo $line_num | awk '{print $1 - 2}'`
        if($rain_line < 3) then
            set rain_line = $line_num
        endif
        set rain_info = `sed ''"$rain_line"'q;d' $weather_db`
    
        set temp = `echo $line_data | cut -d',' -f 2`
        set dew = `echo $line_data | cut -d',' -f 3`
        set humidity = `echo $line_data | cut -d',' -f 4`
        set visibility = `echo $line_data | cut -d',' -f 6`
        set wind = `echo $line_data | cut -d',' -f 8`
        set events = `echo $line_data | cut -d',' -f 11`
        set conditions = `echo $line_data | cut -d',' -f 12`
        set rain_conditions = `echo $rain_info | cut -d',' -f 12`
    
        set temp_var = `echo $visibility | awk '{print int($1)}'`
    
        if("Calm" == $wind) then
            set wind = 0
        endif
    
        if( $temp_var < 0 ) then
            set visibility = 10
        endif
    
        set thunder = "0"
        set rain = "0"
        set cloud = "0"
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
            set cloud = "1"
        endif
    
        set temp_var = `echo $conditions | grep "Partly Cloudy"`
        if( "xxx" != "xxx$temp_var") then
            set cloud = "2"
        endif
    
        set temp_var = `echo $conditions | grep "Mostly Clouds"`
        if( "xxx" != "xxx$temp_var") then
            set cloud = "3"
        endif
    
        set temp_var = `echo $conditions | grep "Overcast"`
        if( "xxx" != "xxx$temp_var") then
            set cloud = "4"
        endif
    
        set temp_var = `echo $rain_conditions | grep "Rain"`
        if( "xxx" != "xxx$temp_var") then
            set rain = "2"
        endif
    
        set temp_var = `echo $rain_conditions | grep "Light Rain"`
        if( "xxx" != "xxx$temp_var") then
            set rain = "1"
        endif
    
        set temp_var = `echo $rain_conditions | grep "Heavy Rain"`
        if( "xxx" != "xxx$temp_var") then
            set rain = "3"
        endif
    
        set temp_var = `echo $conditions | grep "Thunderstorm"`
        if( "xxx" != "xxx$temp_var") then
            set thunder = "2"
        endif
    
        set temp_var = `echo $conditions | grep "Light Thunderstorm"`
        if( "xxx" != "xxx$temp_var") then
            set thunder = "1"
        endif
    
        set temp_var = `echo $conditions | grep "Heavy Thunderstorm"`
        if( "xxx" != "xxx$temp_var") then
            set thunder = "3"
        endif
    
        set sun_data = "SunMoonDB/${year}/${month}/sun_1.info"
        echo $sun_data
    
        # set line_num = `echo $day | awk '{print 116 + ($1 - 1)*3 + 1}'`
        set line_num = `grep -n "^$day  " $sun_data | awk '{print $1 - 1}'`
        if ("xx$line_num" == "xx") then
            echo $location $sun_data $day $lines[$i]
        endif
#         if( $day == 31 ) then
#             @ line_num += 1
#         endif
#         echo $line_num
    
        set line_data = `sed ''"$line_num"'q;d' $sun_data`
#         echo $line_data
    
        set sunrise = $line_data[1]
        set sunrise_h = `echo $sunrise | cut -d':' -f 1 | awk '{print 1*$1}'`
        set sunrise_m = `echo $sunrise | cut -d':' -f 2 | awk '{print 1*$1}'`
        set sunrise = `echo $sunrise_h $sunrise_m | awk '{print $1 + $2/60.0}'`
    #echo $sunrise
    
        set sunset = $line_data[2]
        set sunset_h = `echo $sunset | cut -d':' -f 1 | awk '{print 1*$1}'`
        set sunset_m = `echo $sunset | cut -d':' -f 2 | awk '{print 1*$1}'`
        set sunset = `echo $sunset_h $sunset_m | awk '{print $1 + $2/60.0}'`
    
#         echo $#line_data
        set idx = $#line_data
        set sunpeak = $line_data[$idx]
        set sunpeak_h = `echo $sunpeak | cut -d':' -f 1 | awk '{print 1*$1}'`
        set sunpeak_m = `echo $sunpeak | cut -d':' -f 2 | awk '{print 1*$1}'`
        set sunpeak = `echo $sunpeak_h $sunpeak_m | awk '{print $1 + $2/60.0}'`
    
    
        set img_time = `echo $hour $mins | awk '{print $1 + $2/60.0}'`
    
        set time1 = `echo $img_time $sunrise | awk '{ print $1 - $2}'`
    #    set time1 = `echo $img_time $sunrise | awk '{ print ((($1 - $2) >= 0) ? $1-$2 : $2-$1)}'`
    #     set time1 = `echo $time1 | awk '{print ($1 > 1) ? 1 : $1}'`
    #     set time1 = `echo $time1 | awk '{print ($1 < -1) ? -1 : $1}'`
    #
        set time2 = `echo $img_time $sunset | awk '{ print $1 - $2}'`
    #    set time2 = `echo $img_time $sunset | awk '{ print ((($1 - $2) >= 0) ? $1-$2 : $2-$1)}'`
    #    set time2 = `echo $time2 | awk '{print ($1 > 1) ? 1 : $1}'`
    #    set time2 = `echo $time2 | awk '{print ($1 < -1) ? -1 : $1}'`
    #
        set time3 = `echo $img_time $sunpeak | awk '{ print $1 - $2 }'`
    #    set time3 = `echo $img_time $sunpeak | awk '{ print ((($1 - $2) >= 0) ? $1-$2 : $2-$1)}'`
    #    set time3 = `echo $time3 | awk '{print ($1 > 7 ) ? 7 : $1}'`
    #    set time3 = `echo $time3 | awk '{print ($1 < -7 ) ? -7 : $1}'`
    
        set temp = `echo $temp | awk '{print $1/100.0}'`
        set dew = `echo $dew | awk '{print $1/100.0}'`
        set humidity = `echo $humidity | awk '{print $1/100.0}'`
    #    set visibility = `echo $visibility | awk '{print $1/10.0}'`
    
        set month = `echo $month | awk '{print (((6 - $1) > 0) ? (6 - $1) : ($1 - 6))}'`
    
    # echo "${temp}:${dew}:${humidity}:${visibility}:${wind}:${thunder}:${rain}:${cloud}:${haze}:${mist}:${time1}:${time2}:${time3}" >> $file_name
        
    # echo "${temp} ${dew} ${humidity} ${visibility} ${wind} ${thunder} ${rain} ${cloud} ${haze} ${mist} ${time1} ${time2} ${time3}" >> $file_name
        echo "${time1} ${time2} ${time3} ${visibility} ${cloud} ${haze} ${rain} ${thunder} ${month} ${temp} ${dew} ${mist} ${humidity}" >> $file_name
    
        @ i = ($ix + 1)
    
    end
    
    popd # ]] dbpath

end    
popd
