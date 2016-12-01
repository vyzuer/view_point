#!/bin/csh -f

if($#argv != 1) then
    echo "pass data set path"
    exit
endif

set dbPath = argv[1]

set labels_file = "labels.list"
set img_file = "image.list"

pushd dbPath # [[ dbPath

set file_list = `cat $img_file`
set clusterID = `cat $labels_file`

set j = 1
set i = 1
while($j < $#clusterID)
    set label = $clusterID[$j]
    echo $label
    
    set img_name = $file_list[$i]
    @ i += 1
    set num_seg = $file_list[$i]
    
    set k = 1
    while($k <= $num_seg)
        set filename = "${dbPath}/SegDB/${img_name}/${k}.png"
        ln -sf $filename "${label}/${img_name}_${k}.png"
        @ k++
    end
    @ j++
    @ i++

end
echo $num_of_clusters

popd  # ]] dbPath

