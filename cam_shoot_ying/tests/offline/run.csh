#!/bin/csh 

set dataset_path = /mnt/project/Datasets/DPChallenge/
set dump_path = /mnt/project/VP/cam_shoot_ying/quality_dumps/
set clf_dump_path = /mnt/project/VP/cam_shoot_ying/quality_dumps/classifier/

# extract features for DPChallenge dataset
# python extract_features.py ${dataset_path} ${dump_path}

# python train_classifier.py ${dump_path} ${clf_dump_path}


# extract features for view-point dataset
#
set dataset_path = /home/vyzuer/View-Point/DataSet-VPF/
set dump_path = /mnt/project/VP/cam_shoot_ying/quality_dumps/VP-DUMPS/

: <<'my_END'
foreach location (arcde colognecathedral merlion taj vatican leaningtower liberty indiagate gatewayofindia eifel forbiddencity tiananmen)
    echo $location
    # dump user-ids for each photo, do this just once 
    pushd ${dataset_path}${location} > /dev/null
    cut -d'@' -f 1 image.list > owners.list
    cut -d' ' -f 4 images.details > view.list
    cut -d' ' -f 5 images.details > favs.list
    cut -d' ' -f 9 images.details | cut -d'-' -f 1 > year.list
    popd > /dev/null

end

'my_END'

foreach location (arcde colognecathedral merlion taj vatican leaningtower liberty indiagate gatewayofindia eifel forbiddencity tiananmen)
    echo $location
    # python extract_features.py ${dataset_path}/${location}/ ${dump_path}/${location}/

    # python predict_a_score.py ${clf_dump_path} ${dump_path}${location}/

    # build GMM model and dump for recommendation later on
#     python perform_modeling.py ${dataset_path}${location}/ ${dump_path}${location}/

    # compute popularity of viewpoints for each of the component
    python compute_popularity.py ${dataset_path}${location}/ ${dump_path}${location}/

end


