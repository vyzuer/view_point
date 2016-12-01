#!/bin/csh -f

# set segments_path = "/mnt/windows/Project/DUMPS/offline/merlionImages/"
# set dump_path = "/mnt/windows/Project/DUMPS/offline/merlionImages/cluster_dump/"

set g_dump_path = /mnt/windows/Project/DUMPS/offline/

foreach location (eifel)
# foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia indiagate leaningtower liberty merlion vatican)
# foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia eifel)

    set segments_path = "${g_dump_path}${location}/"
    set dump_path = "${g_dump_path}${location}/cluster_dump/"

    rm -rf $dump_path
    
    set file_name = "${dump_path}km_segments.list"
    set file_name_p = "${dump_path}segments.list"
    
    echo "Merging segmented visual words for clustering..."
    ./combineVisualWords.csh $segments_path $dump_path
    
    set num_clusters = 200
    echo "Cluster visual words..."
    python cluster.py $dump_path $file_name $file_name_p $num_clusters
    
    echo "creating clusters..."
    ./createSegmentClusters.csh $dump_path 
    
    echo "gmm modeling for each segment..."
    python gmm.py $dump_path

end

