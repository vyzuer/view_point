#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/
# set db_path = /mnt/windows/DataSet-VPF2/

pushd $db_path

# foreach location (${db_path}/*)
# foreach location (arcde eifel indiagate liberty taj colognecathedral gatewayofindia leaningtower merlion vatican)
foreach location (forbiddencity tiananmen)
# foreach location (colognecathedral gatewayofindia leaningtower merlion vatican)
    
    pushd $location
    echo $location
    set lines = `cat photo.url.0`

    mkdir -p ImageDBInfo
    pushd ImageDBInfo
#     pushd ${db_path1}${location}/ImageDBInfo

    set i = 7
    set count = 0
    while($i <= $#lines)
        set owner = $lines[$i]
        @ i++
        set photo_id = $lines[$i]
        @ i+=4
        set url = $lines[$i]
        echo $url
        set secret = `echo $url | cut -d"/" -f 5 | cut -d'_' -f 2`
        set newNameFav = "${owner}_${photo_id}.fav"
        set newNameCom = "${owner}_${photo_id}.com"
        set newNameInfo = "${owner}_${photo_id}.info"
        set newNameExif = "${owner}_${photo_id}.exif"

        flickcurl photos.getFavorites $photo_id 50 1 > $newNameFav
        sleep 0.2
#         flickcurl photos.getFavorites $photo_id 50 2 | tee -a $newNameFav
#         flickcurl photos.getFavorites $photo_id 50 3 | tee -a $newNameFav
#         flickcurl photos.getFavorites $photo_id 50 4 | tee -a $newNameFav

#         flickcurl photos.comments.getList $photo_id | tee $newNameCom
        flickcurl photos.getExif $photo_id $secret > $newNameExif
        sleep 0.2
        flickcurl photos.getInfo $photo_id $secret > $newNameInfo
        sleep 0.2
        @ i++
        @ count++
        if ($count > 10) then
            sleep 2
            @ count = 0
        endif
    end

    sleep 5

    popd
    popd
end

popd 

