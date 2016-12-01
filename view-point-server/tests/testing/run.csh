#!/bin/csh 

set dump_path1 = /mnt/project/VP/datta_aesthetics/VP-DUMPS/
set dump_path2 = /mnt/project/VP/cam_shoot_ying/quality_dumps/VP-DUMPS/
set ascore_file = ying_ascore.list

set score_file = cr.log

rm -rf ${score_file}

foreach location (taj colognecathedral gatewayofindia leaningtower merlion arcde eifel indiagate liberty vatican forbiddencity tiananmen)
# foreach location (arcde eifel indiagate liberty vatican)
# foreach location (taj colognecathedral gatewayofindia leaningtower vatican)
# foreach location (merlion arcde eifel indiagate liberty )
# foreach location (forbiddencity tiananmen)
    echo $location

    python evaluate.py ${dump_path1}${location}/ ${dump_path2}/${location}/ ${ascore_file} | tee -a ${score_file}
    
end

python find_average_score.py ${score_file} | tee -a ${score_file}

