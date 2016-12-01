#!/bin/csh -fv

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path

# foreach location (${db_path}/*)
foreach location (forbiddencity tiananmen)

    pushd $location
    echo $location

    set image_details = "images.details"
    set file_name = "geo.info"
    set lines = `cat $image_details`
    
    unlink $file_name
    
    set i = 14
    
    while($i < $#lines)
    
        echo $lines[$i]
        set ix = `echo $i | awk '{print $1+12}'`
        set x1 = `echo $i | awk '{print $1+1}'`
        set x2 = `echo $i | awk '{print $1+2}'`

        set lat = $lines[$x1]
        set long = $lines[$x2]
    
        echo "$lat $long" >> $file_name
    
        @ i = ($ix + 1)
    
    end
    
    popd # ]] dbpath

end    

popd

