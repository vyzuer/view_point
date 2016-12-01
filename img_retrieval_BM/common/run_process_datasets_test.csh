#!/bin/csh 


set db_path = /mnt/windows/DataSet-VP/merlion/
set dump_path = /home/vyzuer/DUMPS/test/merlion/

# set db_path = /mnt/windows/Project/Flickr-YsR/merlionImages/
# set dump_path = /mnt/windows/Project/DUMPS/offline/merlionImages/

python process_datasets.py $db_path $dump_path

