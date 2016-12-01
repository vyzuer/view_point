#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path

# foreach location (${db_path}/*)
foreach location (forbiddencity tiananmen)
# set location = "${db_path}/${i}"
    pushd $location
    echo $location

    set lines = `cat photo.info`

    # pushd ../../../DB/merlionImages
    # pushd /home/yogesh/Project/Flickr-YsR/merlionImages/
    set j = 1
    set i = 16
    # echo $#lines
    echo "name lat long views favorite exposure aperture ISO date time focal focal-2 flash" > images.details
    #while($i <= 100)
    while($i < $#lines)
        set ix = `echo $i | awk '{print $1+14}'`
        
        set allInfo = `echo $lines[$i-$ix]`
        # echo $allInfo
        set username = `echo $allInfo | cut -d" " -f 1`
        set required_fields = `echo $allInfo | cut -d" " -f 1-12 | grep "null"`
        set xname = `echo $allInfo | cut -d" " -f 6 | cut -d"/" -f 5`
        set filename = "${username}_$xname"
        # echo $filename
        @ i+=2
        set lat = $lines[$i]
        @ i++
        set long = $lines[$i]
        @ i++
        set views = $lines[$i]
        @ i+=2
        @ ixx = `echo $i | awk '{print $1+8}'`
        set misc = `echo $lines[$i-$ixx]`
    
        set image_src = "ImageDB/$filename"
        set size = `ls -l $image_src | awk '{print $5}'`
        if(! -z $image_src && $size>50000 && "x$required_fields" == "x" ) then
            echo "$filename $lat $long $views $misc" >> images.details
        else
            echo $image_src
        endif
    
        @ i = ($ix + 1)
    
    end
    popd
end

popd
