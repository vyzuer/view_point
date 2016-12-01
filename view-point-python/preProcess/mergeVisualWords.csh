#!/bin/csh -f
# dataset path, weather data path

set db_path = "/home/scps/myDrive/Flickr-YsR/merlionImages/DB2April"
set dump_path = "/home/scps/myDrive/Copy/Flickr-code/DBR/clustering_904/"

set sal_file = "saliency.list"
set feature_file = "feature.list"

set seg_file = "${dump_path}segments.list"
set img_file = "${dump_path}image.list"
set sal_file2 = "${dump_path}${sal_file}"

pushd $dump_path  # [[ dbpath
mkdir -p SegDB

unlink $seg_file
unlink $img_file
unlink $sal_file2

foreach file ($db_path/*)
    if (-d $file) then
        echo $file
        set img_name = `echo $file | sed 's:'"$db_path/"'::'`
        
        set s_file = "${file}/${sal_file}"
        set segments_path = "${file}/segments/"
        set f_file = "${file}/${feature_file}"
        
        set num_seg = `wc -l $f_file`
        
        set sal_lines = `cat $s_file`

        mkdir -p SegDB/${img_name}

        set i = 1
        foreach line ("`cat $f_file`")
            echo $img_name >> $img_file
            echo $line >> $seg_file
            echo $sal_lines[$i] >> $sal_file2

            ln -sf ${segments_path}/${i}.png SegDB/${img_name}/

            @ i += 1
        end

    endif
end
popd
