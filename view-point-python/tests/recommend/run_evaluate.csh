#!/bin/csh 

set db_path = /mnt/windows/DataSet-VPF/
set res_dump = /mnt/data/DUMPS/DataSet-VPF/
set features_path = /home/vyzuer/DUMPS/visual_words/
# set dump_path = /home/vyzuer/DUMPS/DataSet-VPF-Test5/
set dump_path = /mnt/data/DUMPS/DataSet-VPF-Test5/

foreach location (taj colognecathedral gatewayofindia leaningtower merlion arcde eifel indiagate liberty vatican forbiddencity tiananmen)
# foreach location (arcde eifel indiagate liberty vatican)
# foreach location (taj colognecathedral gatewayofindia leaningtower vatican)
# foreach location (merlion arcde eifel indiagate liberty )
# foreach location (forbiddencity tiananmen)
    echo $location

    # extract features for all the images for later use. one time
    # python extract_features.py ${db_path}/${location}/ ${features_path}/${location}/

    python evaluate.py ${features_path}/${location}/ ${dump_path}/${location}/ ${db_path}/${location}/ ${res_dump}/${location}/
    
end

