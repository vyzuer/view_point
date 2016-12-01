#!/bin/csh -f
# dataset path, weather data path

# ./genWeatherData.csh ~/Project/Flickr-YsR/merlionImages/ ~/Copy/Flickr-code/weatherDB/ /home/yogesh/Copy/Flickr-code/PhotographyAssistance/cameraParameters/DB/

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /home/vyzuer/View-Point/DataSet-VPF/
pushd $db_path

# set loc = $argv[1]

# foreach location (merlion)
foreach location (arcde colognecathedral gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican forbiddencity tiananmen)
# foreach location (liberty merlion taj vatican)

    pushd $location
    echo $location

	set year = 2012
	set month = 1

    while($month <= 12)    
        set sun_data = "SunMoonDB/${year}/${month}/sun_1.info"
        echo $sun_data

        mkdir -p "SunMoonDB/global/"
        set file_name = "SunMoonDB/global/sun_${month}.info"
        echo "sunrise sunset sunpeak" > $file_name
        
		set day = 1

    	while($day <= 31)

            set line_num = `grep -n "^$day  " $sun_data | awk '{print $1 - 1}'`
            if ("xx$line_num" == "xx") then
                echo $location $sun_data $day 
                @ day = ($day + 1)
                continue
            endif
        
            set line_data = `sed ''"$line_num"'q;d' $sun_data`
        
            set sunrise = $line_data[1]
            set sunrise_h = `echo $sunrise | cut -d':' -f 1 | awk '{print 1*$1}'`
            set sunrise_m = `echo $sunrise | cut -d':' -f 2 | awk '{print 1*$1}'`
            set sunrise = `echo $sunrise_h $sunrise_m | awk '{print $1 + $2/60.0}'`
        
            set sunset = $line_data[2]
            set sunset_h = `echo $sunset | cut -d':' -f 1 | awk '{print 1*$1}'`
            set sunset_m = `echo $sunset | cut -d':' -f 2 | awk '{print 1*$1}'`
            set sunset = `echo $sunset_h $sunset_m | awk '{print $1 + $2/60.0}'`
        
            set idx = $#line_data
            set sunpeak = $line_data[$idx]
            set sunpeak_h = `echo $sunpeak | cut -d':' -f 1 | awk '{print 1*$1}'`
            set sunpeak_m = `echo $sunpeak | cut -d':' -f 2 | awk '{print 1*$1}'`
            set sunpeak = `echo $sunpeak_h $sunpeak_m | awk '{print $1 + $2/60.0}'`
        
        
            echo "${sunrise} ${sunset} ${sunpeak}" >> $file_name
        
            @ day = ($day + 1)
        end
        @ month = ($month + 1)
    
    end
    
    popd # ]] dbpath

end    
popd
