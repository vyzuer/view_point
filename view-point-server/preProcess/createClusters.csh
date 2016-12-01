#!/bin/csh -fv

set labels_file = "labels.list"
set dbPath = "/media/Data/Flickr-YsR/merlionImages/"
set image_details = "images.details"
set num_cluster_file = '_num_clusters.info'

pushd dbPath # [[ dbPath

set num_of_clusters = `cat $num_cluster_file`

set lines = `cat $image_details`
set clusterID = `cat $labels_file`

mkdir -p clusters
pushd clusters  # [[ clusters

set i = 0
while ($i < $num_of_clusters)
    mkdir -p $i
    pushd $i  # [[ $i
        mkdir -p ImageDB
        unlink image.list
        echo "name lat long views favorite exposure aperture ISO date time focal shutter-speed" > images.details
    popd  # ]] $i
    @ i++
end

set j = 1
set i = 13

while($j < $#clusterID)
    set label = $clusterID[$j]
    echo $label

    set ix = `echo $i | awk '{print $1+11}'`
    
    set allInfo = `echo $lines[$i-$ix]`
    echo $allInfo

    set filename = $lines[$i]
    echo $filename

    echo "$allInfo" >> ${label}/images.details
    echo "$filename" >> ${label}/image.list
    ln -sf ${dbPath}ImageDB/$filename ${label}/ImageDB/
    @ i = ($ix + 1)
    @ j++

end
echo $num_of_clusters
popd  # ]] clusters

popd  # ]] dbPath
