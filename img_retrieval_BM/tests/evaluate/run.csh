#!/bin/csh 

set base_path = /mnt/project/VP/ODB/
set dataset = ZuBuD
set dataset = Holidays

set db_path = ${base_path}${dataset}/
set dump_path = ${db_path}DUMP/
set model_path = ${dump_path}landmark_objects/
set features_path = ${dump_path}visual_words/

set qdb_path = ${db_path}QueryDB/
set qdump_path = ${qdb_path}DUMP/
set qfeatures_path = ${qdump_path}/visual_words/

# perform image retrieval and evaluate the retrieval step
python evaluate.py ${db_path} ${features_path} ${qfeatures_path}

