#!/bin/csh 


set cluster_dump = /home/vyzuer/DUMPS/landmark_objects.0/merlion/
# set cluster_dump = /home/vyzuer/DUMPS/test/DB2/
set dump_path = /home/vyzuer/DUMPS/test/DB2/

python perform_modeling.py $cluster_dump $dump_path


