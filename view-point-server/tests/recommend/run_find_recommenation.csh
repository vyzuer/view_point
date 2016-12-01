#!/bin/csh 

set db_path = /mnt/windows/DataSet-VPF-Test5/
set model_path = /home/vyzuer/DUMPS/landmark_objects/
# set model_path = /mnt/data/DUMPS/landmark_objects/
set dump_path = /mnt/data/DUMPS/DataSet-VPF-Test5/

# foreach location (merlion arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower vatican)
# foreach location (taj colognecathedral gatewayofindia leaningtower vatican)
# foreach location (merlion arcde eifel indiagate liberty )
# foreach location (arcde leaningtower)
foreach location (forbiddencity tiananmen)
    echo $location
    set rec_type = "gmm_all"
    set gp_filter = 'True'
    python find_recommendation.py ${db_path}/${location}/ ${model_path}/${location}/ ${dump_path}/${location}/ $rec_type $gp_filter
    
end

