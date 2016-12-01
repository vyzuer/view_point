#!/bin/csh 


set db_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/
set dump_path = /home/vyzuer/DUMPS/test/DB2/

python detect_landmark_objects.py $db_path $dump_path

# dump visual words for visualization
./dump_visual_words.csh $dump_path

