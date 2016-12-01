#!/bin/csh -f

set lines = `cat merlion.list`

# pushd ../../../DB/merlionImages
pushd /home/yogesh/Project/Flickr-YsR/merlionImages/
set j = 1
set i = 1
echo $#lines
echo "name lat long views favorite exposure aperture ISO date time focal focal-2 flash" > images.details
#while($i <= 100)
while($i < $#lines)
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
    @ ixx = `echo $i | awk '{print $1+8}'`
    set misc = `echo $lines[$i-$ixx]`

    set folder_num = `echo $allInfo | cut -d"/" -f 7`
    set image_src = "../ImageDB/${folder_num}/$filename"
    set size = `ls -l $image_src | awk '{print $5}'`
    if(! -z $image_src && $size>50000) then
        echo "$filename $lat $long $views $misc" >> images.details
#        cp ../ImageDB/${folder_num}/$filename ImageDB/
    endif

    @ i = ($ix + 1)

end
popd


