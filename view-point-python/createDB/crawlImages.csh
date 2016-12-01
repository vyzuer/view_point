#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF2/

pushd $db_path

foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)

    pushd $location
    echo $location
    ./crawlImages.csh
    popd
end
popd
