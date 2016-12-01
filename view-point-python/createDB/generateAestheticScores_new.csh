#!/bin/csh -f


set alpha = 0.4
set beta = 0.15
set gamma = 1
set delta = 6
set theta = 5

set alpha_0 = 0.2
set beta_0 = 1

# set dataset_path = /mnt/windows/DataSet-YsR/
set dataset_path = /mnt/windows/DataSet-VPF/

# foreach location (${db_path}/*)
# foreach location (arcde)
foreach location (arcde colognecathedral gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican)

    set db_path = "${dataset_path}${location}/"
    pushd $db_path
    echo $location

    set image_list = "image.list"
    set a_score_1 = "aesthetic.scores.1"
    set a_score_0 = "aesthetic.scores"
    unlink $a_score_0
    unlink $a_score_1
    
    set image_details = "images.details"
    
    set lines = `cat $image_details`
    set num = `wc -l $image_list | cut -d' ' -f 1`
    
    set vm = `grep -v "name" $image_details | awk '{print $4}' | sort -n`
    set fm = `grep -v "name" $image_details | awk '{print $5}' | sort -n`
    set median = `echo $num | awk '{print int($1/2)}'`
    echo $median

    set v_m = $vm[$median]
    set f_m = $fm[$median]
    echo $v_m
    echo $f_m

    set alpha = `echo $v_m | awk '{print 2/(1+log($1+1))}'`
    # set beta = `echo $f_m | awk '{print 1/($1+1)}'`
    # set gamma = `echo $v_m $f_m | awk '{print $1/($2+1)}'`
    echo $alpha 
    echo $beta 
    echo $gamma

    set cnt = 1
    set i = 14
    
    while($i < $#lines)
    
        set ix = `echo $i | awk '{print $1+12}'`
        set v = `echo $i | awk '{print $1+3}'`
        set f = `echo $i | awk '{print $1+4}'`
        set views = $lines[$v]
        set favs = $lines[$f]
#         echo $views
#         echo $favs
#         echo $num
#         echo $cnt
        
        set filename = $lines[$i]
        # echo $filename
        
        # set score = `echo $favs $alpha $views $beta $gamma | awk '{print 1 - exp(-1*($1*$2 + $3*$4 + $5))}'`
        #                     1      2      3     4     5      6     7    8     9
        set score_1 = `echo $alpha $views $beta $favs $gamma $delta $num $cnt $theta| awk '{print 1/(1 + exp(-1*($1*log($2+1) + $3*$4 + $5*$4/(log($2+1)+1) + $6*($7 - $8 + 1)/$7 - $9)))}'`
        set score_0 = `echo $alpha_0 $views $beta_0 $favs $num $cnt | awk '{print (1 - exp(-1*($1*$2 + $3*$4)))*exp(-1*($6/$5))}'`
        # echo $score
        echo $score_0 >> $a_score_0
        echo $score_1 >> $a_score_1
        @ i = ($ix + 1)
        @ cnt += 1
    
    end
    
    popd # ]] dbpath

    # plot histogram
    # python plot_a_distib.py ${db_path}

end

