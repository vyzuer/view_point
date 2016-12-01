#!/bin/csh -f

if($#argv != 2 ) then
    echo "Usage: $0 1-dataset path 2-dump_path"
    exit()
endif

set dbPath = $argv[1]
set dump_path = $argv[2]

pushd $dbPath  # [[ dbpath

echo "$0 dumping camera parameters..."

set image_details = "images.details"
set file_name = "${dump_path}/camera.settings"
set lines = `cat $image_details`

unlink $file_name

set i = 14

while($i < $#lines)

    set ix = `echo $i | awk '{print $1+12}'`
    set s = `echo $i | awk '{print $1+5}'`
    set a = `echo $i | awk '{print $1+6}'`
    set ii = `echo $i | awk '{print $1+7}'`
    set f = `echo $i | awk '{print $1+11}'`
    set flash_ = `echo $i | awk '{print $1+12}'`

    set ss = $lines[$s]
    set ap = $lines[$a]
    set iso = $lines[$ii]
#     set ss2 = $lines[$ix]
    set fl = $lines[$f]
    set flash = $lines[$flash_]

#     if("null" == $ss) then
#         set ss = $ss2
#     endif
    
    set ss1 = `echo $ss | cut -d'/' -f 1 -s`
    set ss2 = `echo $ss | cut -d'/' -f 2 -s`

    if("xxx" != "xxx$ss2") then
        set ss = `echo $ss1 $ss2 | awk '{print 1.0*$1/$2}'`
    endif

    set ap1 = `echo $ap | cut -d'/' -f 1 -s`
    set ap2 = `echo $ap | cut -d'/' -f 2 -s`

    if("xxx" != "xxx$ap2") then
        set ap = `echo $ap1 $ap2 | awk '{print 1.0*$1/$2}'`
    endif

    if($iso == 0) then
        set iso = "null"
    endif

    if($ss == 0) then
        set ss = "null"
    endif
    
    if("null" == $fl) then
        set fl = 35.0
    endif

    set fl1 = `echo $fl | cut -d'/' -f 1 -s`
    set fl2 = `echo $fl | cut -d'/' -f 2 -s`

    if("xxx" != "xxx$fl2") then
        set fl = `echo $fl1 $fl2 | awk '{print 1.0*$1/$2}'`
    endif

    
#     set ss = `echo $ss | awk '{print ($1 > 5)?5:$1}'`

    echo $ss $ap $iso $fl $flash >> $file_name

    @ i = ($ix + 1)

end

popd # ]] dbpath
