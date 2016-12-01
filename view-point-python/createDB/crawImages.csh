#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path

foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)

    pushd $location
    echo $location
    set lines = `cat photo.url`

    mkdir -p ImageDB
    pushd ImageDB

    set i = 7
    while($i <= $#lines)
        set owner = $lines[$i]
        @ i+=5
        set url = $lines[$i]
        echo $url
        set lastName = `echo $url | cut -d"/" -f 5`
        set newName = "${owner}_${lastName}"
        echo $newName

        wget -t 5 $url -O $newName
        sleep 0.01
        @ i++
    end

    sleep 2

    popd
    popd
end
popd
