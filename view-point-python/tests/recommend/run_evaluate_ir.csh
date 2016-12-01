#!/bin/csh 

set db_path = /home/vyzuer/View-Point/DataSet-VPF/
set res_dump = /home/vyzuer/View-Point/DUMPS.1/DataSet-VPF-Test5/
set features_path = /home/vyzuer/View-Point/DUMPS.2/visual_words/
set dump_path = /mnt/project/VP/testset/

set location = merlion

python evaluate_ir.py ${features_path}/${location}/ ${dump_path}/${location}/ ${db_path}/${location}/ ${res_dump}/${location}/
    

