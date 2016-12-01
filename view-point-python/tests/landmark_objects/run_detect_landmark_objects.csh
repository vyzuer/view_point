#!/bin/csh 

set db_path = /home/vyzuer/View-Point/DataSet-VP/
set dump_path = /home/vyzuer/View-Point/DUMPS/landmark_objects/

foreach location (arcde colognecathedral taj vatican leaningtower liberty indiagate gatewayofindia eifel forbiddencity tiananmen)
# foreach location (merlion arcde colognecathedral taj vatican leaningtower liberty indiagate gatewayofindia eifel forbiddencity tiananmen)
# foreach location (arcde taj vatican leaningtower liberty )
# foreach location (forbiddencity tiananmen)
# foreach location (vatican)
    echo $location
    python detect_landmark_objects.py ${db_path}/${location}/ ${dump_path}/${location}/
    
    # dump visual words for visualization
    ./dump_visual_words.csh ${dump_path}/${location}/
    
end

