#!/bin/csh -f

set alpha = 0.003
set beta = 0.04
set gamma = 1
set delta = 0.00012
set kappa = 6

# set dataset_path = /mnt/windows/DataSet-YsR/
set dataset_path = /mnt/windows/DataSet-VPF/

# foreach location (${db_path}/*)
foreach location (forbiddencity tiananmen)
# foreach location (arcde colognecathedral gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican)

    set db_path = "${dataset_path}${location}/"
    pushd $db_path
    echo $location

    set image_list = "image.list"
    set a_score = "aesthetic.scores"
    unlink $a_score
    
    set image_details = "images.details"
    
    set lines = `cat $image_details`
    set num = `wc -l $image_list | cut -d' ' -f 1`

    set cnt = 1
    set i = 14
    
    while($i < $#lines)
    
        set ix = `echo $i | awk '{print $1+12}'`
        set v = `echo $i | awk '{print $1+3}'`
        set f = `echo $i | awk '{print $1+4}'`
        set tidx = `echo $i | awk '{print $1+8}'`
        set image_name = $lines[$i]
        echo $image_name

        set views = $lines[$v]
        set favs = $lines[$f]

        set time_info = $lines[$tidx]
        set year = `echo $time_info | cut -d'-' -f 1 -s`
        set month = `echo $time_info | cut -d'-' -f 2 -s | awk '{print 1*$1}'`
        set date = `echo $time_info | cut -d'-' -f 3 -s | awk '{print 1*$1}'`

        set timelapse = `echo $year $month $date | awk '{print 735171 - (($1-1)*365 + ($2-1)*30 + $3)}'`
    
        #                   1       2     3     4      5     6    7     8       9
        # set score = `echo $alpha $views $beta $favs $gamma $num $cnt $delta $timelapse | awk '{print 1 - exp(-1*($1*$2 + $3*$4 + $5*($6-$7)/$6 - $8*log($9+1)/log(10)))}'`
        set score = `echo $alpha $views $beta $favs $gamma $num $cnt $delta $timelapse $kappa | awk '{print (1 - exp(-1*($1*$2 + $3*$4 + $5*($6-$7)/$6)))*(exp(-1*$8*$9+$10)/(1 + exp(-1*$8*$9+$10)))}'`
        echo $score >> $a_score
        @ i = ($ix + 1)
        @ cnt += 1

    end
    
    popd # ]] dbpath

    python plot_a_distib.py ${db_path}

end

