#!/bin/csh -f

set db_path = /mnt/windows/DataSet-YsR/
# set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path

# foreach location (${db_path}/*)
# foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)
foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican esplanade floatMarina)
# foreach location (forbiddencity tiananmen)

    pushd $location
    echo $location

    set geo_min_max = "geo_minmax.list"

    set lat_min = `cat geo.info | cut -d' ' -f 1 | sort -n | xargs echo | awk '{print $1}'`
    set lat_max = `cat geo.info | cut -d' ' -f 1 | sort -n | xargs echo | awk '{print $(NF-1)}'`

    set lon_min = `cat geo.info | cut -d' ' -f 2 | sort -n | xargs echo | awk '{print $1}'`
    set lon_max = `cat geo.info | cut -d' ' -f 2 | sort -n | xargs echo | awk '{print $(NF-1)}'`

    echo $lat_min $lat_max $lon_min $lon_max > $geo_min_max

    popd # ]] dbpath

end    

popd

