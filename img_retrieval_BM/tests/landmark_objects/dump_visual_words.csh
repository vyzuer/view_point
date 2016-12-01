#!/bin/csh -f

if($#argv != 1) then
    echo "Usage $0 data_set_path"
    exit
endif

set dbPath = $1

set labels_file = "segments/labels.list"
set num_cluster_file = 'segments/_num_clusters.info'
set img_file = "segments/png.list"
set centers_file = "segments/centers.list"
set f_features = "segments/segments.list"

pushd dbPath # [[ dbPath

set num_of_clusters = `cat $num_cluster_file`
set file_list = `cat $img_file`
set clusterID = `cat $labels_file`
# set cluster_centers = `cat $centers_file`

pushd segments
mkdir -p SegClustersPNG
pushd SegClustersPNG
mkdir -p None
set i = 0
while ($i < $num_of_clusters)
    mkdir -p $i
    @ i++
end

popd # ]] SegClustersPNG

mkdir -p VisualWords

# set i = 1
# while($i <= $#cluster_centers)
#     set center_id = $cluster_centers[$i]
#     @ center_id++
#     set filename = $file_list[$center_id]
#     
#     set label = $i
#     @ label--
#     
#     ln -sf $filename "VisualWords/${label}.png"
#     @ i++
# end

set i = 1
while($i <= $#clusterID)
    set filename = $file_list[$i]
    
    set label = $clusterID[$i]
    
    if ( $label == -1 ) then
        ln -sf $filename "SegClustersPNG/None/${i}.png"
    else
        ln -sf $filename "SegClustersPNG/${label}/${i}.png"
    endif
    @ i++
end

popd # ]] segments

popd  # ]] dbPath

