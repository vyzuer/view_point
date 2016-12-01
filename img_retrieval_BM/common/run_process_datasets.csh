#!/bin/csh 

set dataset_path = /mnt/windows/DataSet-YsR/
set dump_path = /home/vyzuer/DUMPS/offline/

# foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia eifel)
foreach location (indiagate leaningtower liberty merlion taj vatican)
    echo $location
    python process_datasets.py "${dataset_path}${location}/" "${dump_path}${location}/"
end

