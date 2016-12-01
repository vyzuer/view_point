#!/bin/bash -f

if [ "$#" -ne 1 ]; then
    echo "Usage $0 data_set_path"
    exit
fi

dbPath = $1

labels_file = "labels.list"
num_cluster_file = '_num_clusters.info'
img_file = "image.list"

pushd dbPath # [[ dbPath

num_of_clusters = `cat $num_cluster_file`
file_list = `cat $img_file`
clusterID = `cat $labels_file`

mkdir -p SegClustersPNG
mkdir -p SegClustersInfo
rm -rf SegClustersPNG/*
rm -rf SegClustersInfo/*
pushd SegClustersPNG  # [[ clusters
mkdir -p None
i = 0
while [ $i -lt $num_of_clusters ]; do
    mkdir -p $i
    ((i++))
done
popd # ]] clusters

pushd SegClustersInfo  # [[ ClusterInfo

i = 0
while [ $i -lt $num_of_clusters ]; do
    mkdir -p $i
    ((i++))
done
popd # ]] ClusterInfo

j = 1
i = 1
while [ $j -lt $#clusterID ]; do
    img_name = $file_list[$i]
    let "i+=1"
    num_seg = $file_list[$i]
    
    k = 1
    while [ $k -le $num_seg ]; do
        label = $clusterID[$j]
    
        filename = "${dbPath}/SegmentsDB/${img_name}/${k}.png"
        if [ $label = -1 ]; then
            ln -sf $filename "SegClustersPNG/None/${img_name}_${k}.png"
        else
            ln -sf $filename "SegClustersPNG/${label}/${img_name}_${k}.png"
        fi
        ((j++))
        ((k++))
    done
    ((i++))

done
# echo $num_of_clusters

popd  # ]] dbPath

