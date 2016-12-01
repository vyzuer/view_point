#!/bin/csh 

set dataset_path = /mnt/project/VP/cam_shoot_ying/quality_dumps/
set dump_path = /mnt/project/VP/cam_shoot_ying/quality_dumps/classifier/

python bin_classify_dump.py ${dataset_path} $dump_path

