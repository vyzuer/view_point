#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path

# foreach location (${db_path}/*)
# foreach location (arcde colognecathedral eifel gatewayofindia indiagate leaningtower liberty merlion taj vatican)
foreach location (forbiddencity tiananmen)

    pushd $location
    echo $location

    set lines = `cat photo.url`
    set t_zone = `cat .time_zone`

    set i = 7
    touch photo.info
    echo "owner_ID photo_ID latitude longitude num_views url favorite exposure aperture ISO date time focal focal2 shutter-speed" > photo.info
    while($i <= $#lines)
        set j = $i
        @ j+= 5
        set origInfo = `echo $lines[$i-$j]`
        set owner = $lines[$i]
        @ i++
        set photo_id = $lines[$i]
        @ i+=4
        set url = $lines[$i]
        set lastName = `echo $url | cut -d"/" -f 5`
        set newName = "${owner}_${lastName}"
        set newNameFav = "${owner}_${photo_id}.fav"
        set newNameInfo = "${owner}_${photo_id}.info"
        set newNameExif = "${owner}_${photo_id}.exif"
        set newNameExif2 = "${owner}_${photo_id}.exif2"
        @ i++

        set fav = 0
        set fav = `grep -c "Found person with ID" ImageDBInfo/${newNameFav}`
        
        # exposure
        set exposure = `grep -m 1 "Shutter Speed\[\]*:" ImageDBInfo/${newNameExif2} | cut -d":" -f 2 | cut -c 2-`
        if ( "x$exposure" == "x" ) then
            set exposure = `grep -m 1 "label 'Exposure'" ImageDBInfo/${newNameExif} | cut -d"'" -f 4`
        endif

        # Aperture
        set aperture = `grep -m 1 "Aperture\[\]*:" ImageDBInfo/${newNameExif2} | cut -d":" -f 2 | cut -c 2-`
        if ( "x$aperture" == "x" ) then
            set aperture = `grep -m 1 "label 'Aperture'" ImageDBInfo/${newNameExif} | cut -d"'" -f 4`
        endif

        # ISO
        set iso = `grep -m 1 "ISO\[\]*:" ImageDBInfo/${newNameExif2} | cut -d":" -f 2 | cut -c 2-`
        if ( "x$iso" == "x" ) then
            set iso = `grep -m 1 "label 'ISO Speed'" ImageDBInfo/${newNameExif} | cut -d"'" -f 4 | cut -d"," -f 1`
        endif
        set focal = `grep -m 1 "label 'Focal Length'" ImageDBInfo/${newNameExif} | cut -d"'" -f 4 | cut -d" " -f 1`
        set focal2 = `grep -m 1 "label 'Focal Length " ImageDBInfo/${newNameExif} | cut -d"'" -f 4 | cut -d" " -f 1`
        set sspeed = `grep -m 1 "label 'Shutter Speed'" ImageDBInfo/${newNameExif} | cut -d"'" -f 4`

        # Flash
        set flash_ = `grep -m 1 "Flash Fired" ImageDBInfo/${newNameExif2}`
        if ( "x$flash_" == "x" ) then
            set flash_ = `grep -m 1  "label 'Flash'" ImageDBInfo/${newNameExif} | grep -w 'Fired\|On' | grep -v fire `
        else
            set flash_ = `echo $flash_ | grep "True"`
        endif

        set flash = 1
        if ( "x$flash_" == "x" ) then
            set flash = 0
        endif
        
        if ( "x$exposure" == "x" ) then 
            set exposure = nul 
        endif 
        if ( "x$aperture" == "x" ) then 
            set aperture = nul 
        endif
        if ( "x$iso" == "x") then 
            set iso = nul 
        endif
        if ("x$focal" == "x" ) then 
            set focal = nul 
        endif
        if ("x$focal2" == "x" ) then 
            set focal2 = nul 
        endif
        if ("x$sspeed" == "x" ) then 
            set sspeed = nul 
        endif
        if ("x$fav" == "x" ) then 
            set fav = 0 
        endif

        # time
        set date_1 = `grep -m 1 "Date/Time Original" ImageDBInfo/${newNameExif2} | cut -d' ' -f 17`
        set time_1 = `grep -m 1 "Date/Time Original" ImageDBInfo/${newNameExif2} | cut -d' ' -f 18`
        echo $time_1

        if ( "x$time_1" == "x" ) then
            set date_1 = `grep -m 1 "label 'Date and Time (Original)'" ImageDBInfo/${newNameExif} | cut -d"'" -f 4 | cut -d" " -f 1`
            set time_1 = `grep -m 1 "label 'Date and Time (Original)'" ImageDBInfo/${newNameExif} | cut -d"'" -f 4 | cut -d" " -f 2`

            set date_2 = `grep -m 1 "label 'Date and Time (Original)'" ImageDBInfo/${newNameExif} | cut -d"'" -f 6 | cut -d" " -f 1`
            set time_2 = `grep -m 1 "label 'Date and Time (Original)'" ImageDBInfo/${newNameExif} | cut -d"'" -f 6 | cut -d" " -f 2`

            if ( "x$time_2" != "x" && "x$time_2" != "x(null)" ) then
                echo $time_2
                echo $date_2
                set date_1 = $date_2
                set time_1 = $time_2
            endif
            echo $time_1
        endif

        if ( "x$time_1" == "x" ) then
            set date_1 = `grep -m 1 "dates_taken " ImageDBInfo/${newNameInfo} | cut -d"'" -f 2 | cut -d" " -f 1`
            set time_1 = `grep -m 1 "dates_taken " ImageDBInfo/${newNameInfo} | cut -d"'" -f 2 | cut -d" " -f 2`
        endif

        if ("x$time_1" == "x" ) then 
            set time_1 = null
        endif
        if ("x$date_1" == "x") then 
            set date_1 = null 
        endif

        if ( "x$time_1" != "xnull" ) then
            set t_mins = `echo $time_1 | awk -F':' '{print 60*$1 + $2}'`
            set secs = `echo $time_1 | cut -d':' -f 3`
            # three cases Z/+/-
            set z_ = `echo $time_1 | grep 'Z' `
            set plus_ = `echo $time_1 | grep '+' `
            set minus_ = `echo $time_1 | grep '-' `

            set add_mins = $t_zone
            if ( "x$z_" != "x" ) then
                set add_mins = 0
                set secs = `echo $secs | cut -d'Z' -f 1`
            else if ("x$plus_" != "x" ) then
                set add_mins = `echo $time_1 | cut -d'+' -f 2 | awk -F':' '{print 60*$1 + $2}'`
                set secs = `echo $secs | cut -d'+' -f 1`
            else if ("x$minus_" != "x" ) then
                set add_mins = `echo $time_1 | cut -d'-' -f 2 | awk -F':' '{print -(60*$1 + $2)}'`
                set secs = `echo $secs | cut -d'-' -f 1`
            endif

            set t_mins = `echo $t_mins $t_zone $add_mins | awk '{print $1 + $2 - $3}'`

            set time_adj = `echo $t_mins | awk '{print ($1 > 1440) ? $1 - 1440 : ( ($1 < 0) ? 1440 + $1 : $1) }'`
            echo $time_adj
            set hr_n = `echo $time_adj | awk '{ printf("%02d", $1/60)}'`
            set min_n = `echo $time_adj | awk '{ printf("%02d", $1%60)}'`
 
            set time_1 = "${hr_n}:${min_n}:${secs}"
            echo $time_1

        endif

        set date = `echo $date_1 | sed "s/:/-/g"`

#         # correct the time zone
#         set country = `grep "field owner_location" ImageDBInfo/${newNameInfo} | cut -d"'" -f 2 | rev | cut -d"," -f 1 | rev `
#         
#         if ( "x$country" == "x" ) then
#             set country = "Singapore"
#         else if ($country[0] == ' ' ) then
#             set country = `echo $country | cut -c 2-`
#         endif
#         echo $newNameInfo
#         echo $folder
#         echo $country
#         set time_new = `grep -i -m 1 "^$country" $time_zone | cut -d":" -f 2 | cut -d" " -f 2 | awk '{print 60*$1}'`
#         echo $time_new
#         set hrs = `echo $time | cut -d":" -f 1 `
#         set mins_ = `echo $time | cut -d":" -f 2 `
#         set mins = `echo $hrs $mins_ | awk '{print 60*$1 + $2}'`
#         echo $mins
#         set t = `echo $time_new $mins | awk '{print $2 - $1 + 480}'`
#         echo $t
#         set time_adj = `echo $t | awk '{print ($t > 1440) ? $t - 1440 : ( ($t < 0) ? 1440 + $t : $t) }'`
#         echo $time_adj
#         set hr_n = `echo $time_adj | awk '{print int($1/60)}'`
#         set min_n = `echo $time_adj | awk '{print $1%60}'`
# 
#         set n_time = "${hr_n}:${min_n}:00"
#         echo $n_time

        echo "$origInfo $fav $exposure $aperture $iso $date $time_1 $focal $focal2 $flash" >> photo.info

    end

    popd
end

popd
