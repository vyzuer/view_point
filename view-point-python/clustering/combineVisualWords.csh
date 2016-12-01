#!/bin/tcsh -f

# set db_path = "/home/scps/myDrive/Flickr-YsR/merlionImages/DB2April/"
# set dump_path = "/home/scps/myDrive/Copy/Flickr-code/DBR/clustering_1104/"

if($#argv != 2) then
    echo "Usage : $0 segments_path dump_path"
    exit
endif

set db_path = $1
set dump_path = $2

mkdir -p $dump_path

# input

set image_details = "${db_path}/images.details"
set image_list = "${db_path}/image.list"
set a_score_list = "${db_path}/aesthetic.scores"
set weather_info = "${db_path}/weather.info"

set sal_file = "saliency.list"
set km_sal_file = "km_saliency.list"
set feature_file = "feature.list"
set position_list = "pos.list"

# OUTPUT
set seg_file = "${dump_path}segments.list"
set kmeans_seg_file = "${dump_path}km_segments.list"
set img_file = "${dump_path}image.list"
set kmeans_img_file = "${dump_path}km_image.list"
set sal_file2 = "${dump_path}${sal_file}"
set km_sal_file2 = "${dump_path}${km_sal_file}"
set details_file_new = "${dump_path}/images.details"
set pos_file = "${dump_path}/${position_list}"
set time_file = "${dump_path}/time.list"
set geo_list = "${dump_path}/geo.list"
set ascore_list = "${dump_path}/aesthetic.scores"

pushd $dump_path  # [[ dbpath
mkdir -p SegmentsDB

# unlink $seg_file
# unlink $img_file
# unlink $sal_file2
# unlink $details_file_new
# unlink $pos_file
# unlink $time_file
# unlink $geo_list
# unlink $ascore_list

set line_info = `cat $image_details`
set img_list = `cat $image_list`
set w_info = `cat $weather_info`
set a_score = `cat $a_score_list`

set j = 14 # images.details
set i = 1 # image.list
set l = 3 # weather

while($i < $#img_list)
    # echo $i
    set jx = `echo $j | awk '{print ($1+12)}'`
    # echo $jx
    set info = `echo $line_info[$j-$jx]`

    set img_name = `echo "$img_list[$i]" | sed 's/\.jpg//'`
    set file = "${db_path}/segment_dumps/${img_name}"
    # echo $file
    if (-d $file) then
        # set img_name = `echo $file | sed 's:'"$db_path/"'::'`
        
        set s_file = "${file}/${sal_file}"
        set segments_path = "${file}/segments/"
        set f_file = "${file}/${feature_file}"
        set p_file = "${file}/${position_list}"
        
        set num_seg = `wc -l $s_file | cut -d' ' -f 1`
        
        echo "${img_name} ${num_seg}"  >> $img_file
        echo $info >> $details_file_new

        mkdir -p SegmentsDB/${img_name}
        
        cat $f_file >> $seg_file
        head -n $num_seg $s_file >> $sal_file2

        cat $p_file >> $pos_file

        @ i1 = ($j + 1)
        @ i2 = ($j + 2)
        set geo = `echo $line_info[$i1-$i2]`
        set k = 1
        while($k <= $num_seg)
            echo $w_info[$l] >> $time_file
            echo $a_score[$i] >> $ascore_list
            echo $geo >> $geo_list

            set temp_file = "${segments_path}/${k}.png"
            ln -sf $temp_file SegmentsDB/${img_name}/

            @ k += 1
        end

        set good_img = `echo $a_score[$i] | awk '{print int(100*$1)}'`

        # dump files for kmeans clustering
        if ( $good_img > 80 ) then
            cat $f_file >> $kmeans_seg_file
            head -n $num_seg $s_file >> $km_sal_file2
            echo "${img_name} ${num_seg}"  >> $kmeans_img_file
        endif


    endif
    @ j = ($jx + 1)
    @ i += 1
    # dataset taj has different weather info.. make the below step 12 instead of 13
    @ l += 13
    # echo $j
    # echo $jx
end
popd

