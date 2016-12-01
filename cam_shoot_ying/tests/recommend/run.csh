#!/bin/csh 

set dataset_path = /home/vyzuer/View-Point/DataSet-VPF/
set dump_path = /mnt/project/VP/cam_shoot_ying/quality_dumps/VP-DUMPS/
set features_path = /home/vyzuer/View-Point/DUMPS.2/visual_words/
set test_path = /home/vyzuer/View-Point/DataSet-VPF-Test5/

foreach location (arcde colognecathedral merlion taj vatican leaningtower liberty indiagate gatewayofindia eifel forbiddencity tiananmen)
# foreach location (indiagate forbiddencity tiananmen)
    echo $location

    python evaluate.py ${dataset_path}/${location}/ ${dump_path}/${location}/ ${features_path}/${location}/ ${test_path}/${location}/

end

