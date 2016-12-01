#!/bin/csh -f

set db_path = /mnt/windows/DataSet-VPF2/

pushd $db_path

# foreach location (indiagate liberty taj gatewayofindia leaningtower merlion vatican) 
foreach location (arcde leaningtower)
    pushd $location
    echo $location
    ./crawlSunMoonInfo.csh
    popd
end
