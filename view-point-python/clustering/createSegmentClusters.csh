#!/bin/csh -f

if($#argv != 1) then
    echo "Usage $0 data_set_path"
    exit
endif

set dbPath = $1

set labels_file = "labels.list"
set km_labels_file = "km_labels.list"
set num_cluster_file = '_num_clusters.info'
set img_file = "image.list"
set km_img_file = "km_image.list"

set f_a_score = "aesthetic.scores"
set f_geo = "geo.list"
set f_pos = "pos.list"
set f_sal = "saliency.list"
set f_features = "segments.list"
set f_time = "time.list"

pushd dbPath # [[ dbPath

set num_of_clusters = `cat $num_cluster_file`
set file_list = `cat $img_file`
set km_file_list = `cat $km_img_file`
set clusterID = `cat $labels_file`
set km_clusterID = `cat $km_labels_file`

set a_score = `cat $f_a_score`
set geo_list = `cat $f_geo`
set pos_list = `cat $f_pos`
set sal_list = `cat $f_sal`
# set seg_list = `cat $f_features`
set time_list = `cat $f_time`

mkdir -p km_SegClustersPNG
mkdir -p SegClustersPNG
mkdir -p SegClustersInfo
# rm -rf SegClustersPNG/*
# rm -rf SegClustersInfo/*
pushd km_SegClustersPNG  # [[ clusters
mkdir -p None
set i = 0
while ($i < $num_of_clusters)
    mkdir -p $i
    @ i++
end
popd # ]] clusters
pushd SegClustersPNG  # [[ clusters
mkdir -p None
set i = 0
while ($i < $num_of_clusters)
    mkdir -p $i
    @ i++
end
popd # ]] clusters

pushd SegClustersInfo  # [[ ClusterInfo

set i = 0
while ($i < $num_of_clusters)
    mkdir -p $i
    @ i++
end
popd # ]] ClusterInfo

set j = 1
set i = 1

while($j < $#km_clusterID)
    set img_name = $km_file_list[$i]
    @ i += 1
    set num_seg = $km_file_list[$i]
    
    set k = 1
    while($k <= $num_seg)
        set label = $km_clusterID[$j]
    
        set filename = "${dbPath}/SegmentsDB/${img_name}/${k}.png"
        if ( $label == -1 ) then
            ln -sf $filename "km_SegClustersPNG/None/${img_name}_${k}.png"
        else
            ln -sf $filename "km_SegClustersPNG/${label}/${img_name}_${k}.png"
        endif
        @ j++
        @ k++
    end
    @ i++

end

set j = 1
set i = 1

set m = 1  # geo
set l = 1  # features
# set lx = `echo $#seg_list $#clusterID | awk '{print $1/$2}'`

while($j < $#clusterID)
    set img_name = $file_list[$i]
    @ i += 1
    set num_seg = $file_list[$i]
    
    set k = 1
    while($k <= $num_seg)
        set label = $clusterID[$j]
    
#         set filename = "${dbPath}/SegmentsDB/${img_name}/${k}.png"
#         if ( $label == -1 ) then
#             ln -sf $filename "SegClustersPNG/None/${img_name}_${k}.png"
#         else
#             ln -sf $filename "SegClustersPNG/${label}/${img_name}_${k}.png"

        if ( $label != -1 ) then
            # collect other details
            echo $a_score[$j] >> "SegClustersInfo/${label}/${f_a_score}"
            echo $sal_list[$j] >> "SegClustersInfo/${label}/${f_sal}"
            echo $time_list[$j] >> "SegClustersInfo/${label}/${f_time}"

            set mx = `echo $m | awk '{print $1 + 1}'`
            echo $geo_list[$m-$mx] >> "SegClustersInfo/${label}/${f_geo}"
            echo $pos_list[$m-$mx] >> "SegClustersInfo/${label}/${f_pos}"

            # set lxx = `echo $l $lx | awk '{print $1+$2-1}'`
            # echo $seg_list[$l-$lxx] >> "SegClustersInfo/${label}/${f_features}"
        endif
        @ j++
        @ k++
        @ m += 2
        # @ l += $lx
    end
    @ i++

end
# echo $num_of_clusters

popd  # ]] dbPath

