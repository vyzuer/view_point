#!/bin/csh -f

set segments_path = "/home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/dump/"
set dump_path = "/home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/cluster_dump/"

# set segments_path = "/mnt/windows/Project/DUMPS/offline/merlionImages/"
# set dump_path = "/mnt/windows/Project/DUMPS/offline/merlionImages/cluster_dump/"

rm -rf $dump_path

set file_name = "${dump_path}km_segments.list"
set file_name_p = "${dump_path}segments.list"

echo "Merging segmented visual words for clustering..."
./combineVisualWords.csh $segments_path $dump_path

set num_clusters = 20
echo "Cluster visual words..."
python cluster.py $dump_path $file_name $file_name_p $num_clusters

echo "creating clusters..."
./createSegmentClusters.csh $dump_path 

echo "gmm modeling for each segment..."
python gmm.py $dump_path

