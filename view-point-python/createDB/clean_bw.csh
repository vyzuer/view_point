#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-VPF2/
# set script_src = /home/vyzuer/Copy/Flickr-code/createDB/clean_bw_lm.py
set script_src = /home/vyzuer/Copy/Flickr-code/createDB/clean_bw.py
set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path

# foreach location (${db_path}*)
# foreach location (arcde colognecathedral eifel gatewayofindia indiagate leaningtower liberty merlion taj vatican)
foreach location (forbiddencity tiananmen)
    echo $location
    python $script_src $location
end

popd
