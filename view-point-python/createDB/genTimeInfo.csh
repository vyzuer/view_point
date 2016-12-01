#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/

pushd db_path

# foreach location (`ls ${db_path}`)
foreach location (forbiddencity tiananmen)

    pushd $location
    echo $location

    set image_details = "images.details"
    set file_name = "time.info"
    set lines = `cat $image_details`
    
    unlink $file_name
    
    set i = 14
    
    while($i < $#lines)
    
        echo $lines[$i]
        set ix = `echo $i | awk '{print $1+12}'`
        set x1 = `echo $i | awk '{print $1+9}'`
 
        set time_info = $lines[$x1]
    
        echo "$time_info" >> $file_name
    
        @ i = ($ix + 1)
    
    end
    
    popd # ]] dbpath

end    

popd
