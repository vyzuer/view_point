#!/bin/csh 

set dataset_path = /mnt/project/Datasets/DPChallenge/
set dump_path = /mnt/project/VP/datta_aesthetics/
set clf_dump_path = /mnt/project/VP/datta_aesthetics/classifier_640/

# python train_classifier.py ${dump_path} ${clf_dump_path}

# extract features for view-point dataset
set dataset_path = /home/vyzuer/View-Point/DataSet-VPF/
set dump_path = /mnt/project/VP/datta_aesthetics/VP-DUMPS/

foreach location (arcde colognecathedral merlion taj vatican leaningtower liberty indiagate gatewayofindia eifel forbiddencity tiananmen)
    echo $location

    # python predict_a_score.py ${clf_dump_path} ${dump_path}${location}/

end

set dataset_path = /home/vyzuer/View-Point/DataSet-VPF/
set dump_path = /mnt/project/VP/view_direction/VP-DUMPS/
set a_dump_path = /mnt/project/VP/datta_aesthetics/VP-DUMPS/

foreach location (merlion arcde)
    echo $location

    python find_direction.py ${dataset_path}/${location} ${dump_path}${location}/ ${a_dump_path}/${location}/

end
