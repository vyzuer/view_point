#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path

# foreach location (${db_path}/*)
foreach location (forbiddencity tiananmen)

    pushd $location
    echo $location

    set lines = `cat photo.url`

    set i = 7
    while($i <= $#lines)
        set owner = $lines[$i]
        @ i++
        set photo_id = $lines[$i]
        @ i+=4
        set url = $lines[$i]
        set secret = `echo $url | cut -d"/" -f 5 | cut -d'_' -f 2`
        set newNameExif = "ImageDBInfo/${owner}_${photo_id}.exif2"
        set img_name = "${owner}_${photo_id}_${secret}_z.jpg"
        set image_src = ImageDB/$img_name
        set size = `ls -l $image_src | awk '{print $5}'`
        echo $image_src
        if(! -z $image_src && $size>50000) then
#            echo $image_src
#            echo $newNameExif
            exiftool $image_src > $newNameExif
        endif
        @ i++
    end

    @ folder++
    popd
end

popd
