#!/bin/csh 


set db_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/
set dump_path = /home/vyzuer/DUMPS/test/DB2_0/

python dump_visual_segments.py $db_path $dump_path

