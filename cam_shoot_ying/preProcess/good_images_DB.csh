#!/bin/csh -f

set db_path = "/home/yogesh/Copy/Flickr-code/DB/merlionImages/"

pushd $db_path

set lines = `cat images.details`
set map_list = `cat map.list`

mkdir -p Good_Image_DB

pushd Good_Image_DB

set image_details = "images.details"

echo "name lat long views favorite exposure aperture ISO date time focal shutter-speed" > $image_details

mkdir -p ImageDB

set j = 1
set i = 13
while($j < $#map_list)
    set map = $map_list[$j]
    set ix = `echo $i | awk '{print $1+11}'`
    
    if( $map == 0) then
        
        set filename = $lines[$i]
        echo $filename
        set allInfo = `echo $lines[$i-$ix]`
        echo $allInfo >> $image_details
        echo $allInfo
        cp ${db_path}/ImageDB/${filename} ./ImageDB/
    endif

    @ i = ($ix + 1)
    @ j++

end

popd

popd
