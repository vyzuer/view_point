#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path1 = /mnt/windows/DataSet-VPF/
set db_path2 = /mnt/windows/DataSet-VPF2/

foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)
# foreach location (arcde)

    echo $location

    # make a copy of photo.url
    cp ${db_path1}${location}/photo.url ${db_path1}${location}/photo.url.1
    python merge_photo_url.py ${db_path1}${location} ${db_path2}${location}

end

