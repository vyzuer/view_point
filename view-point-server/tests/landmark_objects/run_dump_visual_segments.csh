#!/bin/csh 

set db_path = /mnt/windows/DataSet-VPF/
set dump_path = /home/vyzuer/DUMPS/visual_words/

# foreach location (taj colognecathedral gatewayofindia leaningtower vatican)
# foreach location (merlion arcde eifel indiagate liberty )
foreach location (forbiddencity tiananmen)
    echo $location
    python dump_visual_segments.py ${db_path}/${location}/ ${dump_path}/${location}/
    
end

