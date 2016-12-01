#!/bin/csh -f

set db_path = /mnt/windows/DataSet-YsR/
# set db_path = /mnt/windows/DataSet-VPF/

# foreach location (${db_path}/*)
# foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)
foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican esplanade floatMarina)
# foreach location (forbiddencity tiananmen)

    echo $location
    unlink ${db_path}/${location}/weather.minmax
    python minmax_weather.py ${db_path}/${location}/

end    

