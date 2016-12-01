#!/bin/csh -f

if($#argv != 1 ) then
    echo "pass dataset path"
    exit()
endif

set dbPath = $argv[1]

set alpha = 1.15
set beta = 0.6
set gamma = 20
set delta = 4

pushd $dbPath  # [[ dbpath
# set image_list = "image.list"
set a_score = "aesthetic.scores"
# unlink $image_list
unlink $a_score

set image_details = "images.details"

set lines = `cat $image_details`

set i = 14

while($i < $#lines)

    set ix = `echo $i | awk '{print $1+12}'`
    set v = `echo $i | awk '{print $1+3}'`
    set f = `echo $i | awk '{print $1+4}'`
    set views = $lines[$v]
    set favs = $lines[$f]
    echo $views
    echo $favs
    
    set filename = $lines[$i]
    echo $filename
    
    # set score = `echo $favs $alpha $views $beta $gamma | awk '{print 1 - exp(-1*($1*$2 + $3*$4 + $5))}'`
    set score = `echo $alpha $views $beta $favs $gamma $delta| awk '{print 1/(1 + exp(-1*($1*log($2+1) + $3*$4 + $5*$4/($2+1) - $6)))}'`
    echo $score
    echo $score >> $a_score
# echo "$filename" >> $image_list
    @ i = ($ix + 1)

end

popd # ]] dbpath
