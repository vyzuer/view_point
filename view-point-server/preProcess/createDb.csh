#!/bin/csh -fv

set num_of_clusters = `wc -l center.kmeans | cut -d" " -f 1`
pushd ../../../DB/merlionDB/DB
set i = 0
while ($i < $num_of_clusters)
    mkdir -p $i
    pushd $i
        mkdir -p ImageDB
        mkdir -p InfoDB
        echo "name lat long views favorite exposure aperture ISO date time focal shutter-speed" > InfoDB/images.details2
    popd
    @ i++
end

popd

set lines = `cat merlion.list`
set clusterID = `cat label.kmeans`
pushd ../../../DB/merlionDB/DB
set j = 1
set i = 1
#while($j <= 2)
while($j < $#clusterID)
    set label = $clusterID[$j]
    set ix = `echo $i | awk '{print $1+14}'`
    
    set allInfo = `echo $lines[$i-$ix]`
    echo $allInfo
    set username = `echo $allInfo | cut -d":" -f 2 | cut -d" " -f 1`
    set xname = `echo $allInfo | cut -d" " -f 6 | cut -d"/" -f 5`
    set filename = "${username}_$xname"
    echo $filename
    @ i+=2
    set lat = $lines[$i]
    @ i++
    set long = $lines[$i]
    @ i++
    set views = $lines[$i]
    @ i+=2
    @ ixx = `echo $i | awk '{print $1+7}'`
    set misc = `echo $lines[$i-$ixx]`

    set folder_num = `echo $allInfo | cut -d"/" -f 6`
    echo "$filename $lat $long $views $misc" >> ${label}/InfoDB/images.details
    cp ../../ImageDB/${folder_num}/$filename ${label}/ImageDB/
    @ i = ($ix + 1)
    @ j++

end
popd


