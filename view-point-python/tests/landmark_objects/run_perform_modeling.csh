#!/bin/csh 

set cluster_dump = /home/vyzuer/View-Point/DUMPS/landmark_objects/
# set cluster_dump = /mnt/data/DUMPS/landmark_objects.1/
set dump_path = /home/vyzuer/View-Point/DUMPS/visual_words/

foreach location (arcde colognecathedral gatewayofindia indiagate leaningtower liberty taj vatican eifel forbiddencity tiananmen)
# foreach location (merlion arcde colognecathedral gatewayofindia indiagate leaningtower liberty taj vatican eifel forbiddencity tiananmen)
# foreach location (merlion arcde colognecathedral gatewayofindia indiagate)
# foreach location (leaningtower liberty taj vatican eifel)
# foreach location (forbiddencity tiananmen)
# foreach location (merlion)
    echo $location
    # set gmm_type = "basic"
    foreach type ("basic" "time" "weather")
        set gmm_type = $type
        python perform_modeling.py ${cluster_dump}/${location}/ ${dump_path}/${location}/ $gmm_type
    end
    
end

