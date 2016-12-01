#!/bin/csh -f

set db_path = /mnt/windows/DataSet-VPF/

# foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)
foreach location (forbiddencity templeofheaven)
    echo $location
    python unique_photos.py ${db_path} ${location}
end

