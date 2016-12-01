#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VP/

pushd $db_path

# foreach location (${db_path}/*)
# foreach location (forbiddencity templeofheaven)
foreach location (forbiddencity tiananmen)

    pushd $location
    echo $location
    # mv photo.url photo.url_0
    touch photo.url
    echo "owner_ID photo_ID latitude longitude num_views url" > photo.url
    foreach list (./flickrInfo/*)
    # foreach list (`cat flickr.list`)
    
        set num_photo = `grep "Search result returned" $list | cut -d' ' -f 5`
        set i = 0
        set lineNum = 2
        set formatLen = 18
        while ($i < $num_photo)
            set endLine = `echo $lineNum | awk '{print $1+20}'`
            @ lineNum++
            @ lineNum++
            sed -n $lineNum,${endLine}p $list > __tmp_ysr
            @ lineNum--
            @ lineNum--
            set formatLen = `grep -m 2 -n "Search result photo" __tmp_ysr | cut -d':' -f 1`
            @ formatLen++
            if($formatLen < 10) then
                set formatLen = 18
            endif
#            echo $formatLen
#            echo $lineNum
#            echo $list
            set endLine = `echo $lineNum $formatLen | awk '{print $1+$2}'`
            sed -n $lineNum,${endLine}p $list > __tmp_ysr
            set photo_ID = `cat __tmp_ysr | grep "photo with URI" | cut -d' ' -f 6`
            set farm_ID = `cat __tmp_ysr | grep "field farm" | cut -d"'" -f 2`
            set server_ID = `cat __tmp_ysr | grep "field server" | cut -d"'" -f 2`
            set latitude = `cat __tmp_ysr | grep "field location_latitude" | cut -d"'" -f 2`
            set longitude = `cat __tmp_ysr | grep "field location_longitude" | cut -d"'" -f 2`
            set owner_ID = `cat __tmp_ysr | grep "field owner_nsid" | cut -d"'" -f 2`
            set secret = `cat __tmp_ysr | grep "field secret" | cut -d"'" -f 2`
            set num_views = `cat __tmp_ysr | grep "field views" | cut -d"'" -f 2`

            set url = "https://farm${farm_ID}.staticflickr.com/${server_ID}/${photo_ID}_${secret}_z.jpg"

            echo "$owner_ID $photo_ID $latitude $longitude $num_views $url" >> photo.url

            @ i++
            set lineNum = `echo $lineNum $formatLen | awk '{print $1+$2}'`
        end
    end

    popd

end
popd
