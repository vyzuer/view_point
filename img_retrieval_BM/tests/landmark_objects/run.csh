#!/bin/csh 

set base_path = /mnt/project/VP/ODB/
set dataset = ZuBuD
set dataset = Holidays

set db_path = ${base_path}${dataset}/
set dump_path = ${db_path}DUMP/
set cluster_dump = ${dump_path}landmark_objects/

set qdb_path = ${db_path}QueryDB/
set qdump_path = ${qdb_path}DUMP/

# first detect the landmark objects
python detect_landmark_objects.py ${db_path} ${dump_path}landmark_objects/
    
# dump visual words for visualization
./dump_visual_words.csh ${dump_path}landmark_objects/
    
# dump the features for later use
python perform_feature_extraction.py ${db_path} ${cluster_dump} ${cluster_dump}

# now run for query images
python dump_visual_segments.py ${qdb_path} ${qdump_path}visual_words/

# dump the features for later use
python perform_feature_extraction.py ${qdb_path} ${cluster_dump} ${qdump_path}visual_words/

