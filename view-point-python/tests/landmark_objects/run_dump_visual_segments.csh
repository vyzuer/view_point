#!/bin/csh 

set db_path = /home/vyzuer/View-Point/DataSet-VPF/
set dump_path = /home/vyzuer/View-Point/DUMPS/visual_words/
 
foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower vatican forbiddencity tiananmen)
# foreach location (merlion arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower vatican forbiddencity tiananmen)
# foreach location (merlion arcde eifel indiagate liberty )
# foreach location (forbiddencity tiananmen)
# foreach location (merlion)
    echo $location
    python dump_visual_segments.py ${db_path}/${location}/ ${dump_path}/${location}/
    
end

