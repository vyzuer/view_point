#!/bin/csh -f

set db_path = "/home/yogesh/Project/Flickr-YsR/merlionImages/"

pushd $db_path

set lines = `cat images.details`
set score_list = `cat aesthetic.scores`

mkdir -p top_N_images

pushd top_N_images

set image_details = "images.details"

echo "name lat long views favorite exposure aperture ISO date time focal shutter-speed" > $image_details

mkdir -p ImageDB

set j = 1
set i = 13
while($j < $#score_list)
    set score = $score_list[$j]
    set ix = `echo $i | awk '{print $1+11}'`
    set flag = `echo $score | awk '{print ($0 > 0.9) ? 1 : 0}'`
    if( $flag == 1) then
        
        set filename = $lines[$i]
        echo $filename
        set allInfo = `echo $lines[$i-$ix]`
        echo $allInfo >> $image_details
        # echo $allInfo
        cp -f ${db_path}/ImageDB/${filename} ./ImageDB/
    endif

    @ i = ($ix + 1)
    @ j++

end

popd

popd
