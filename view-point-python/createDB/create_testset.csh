#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/
set dump_path = /mnt/windows/DataSet-VPF-Test5/

# foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)
foreach location (forbiddencity tiananmen)

    echo $location
    python gen_testset_vp.py ${db_path}/${location}/ ${dump_path}/${location}/

    set img_list = "${dump_path}/${location}/image.list"
    set img_dst_path = ${dump_path}/${location}/ImageDB
    set img_src_path = ${db_path}/${location}/ImageDB

    mkdir -p $img_dst_path
    foreach img (`cat $img_list`)
        ln -sf $img_src_path/$img $img_dst_path/
    end

end    

