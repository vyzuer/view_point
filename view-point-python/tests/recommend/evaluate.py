import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python')

import prediction.evaluation as evals

def landmark_objects(dataset_path, dump_path, rec_type, db_path, res_dump, i_filter):
    # evals.process_dataset(dataset_path, dump_path, db_path, res_dump, rec_type=rec_type, i_filter=i_filter, dump_map=False)
    evals.find_geo_density(dataset_path)

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print "Usage : dataset_path dump_path db_path res_dump"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    db_path = sys.argv[3]
    res_dump = sys.argv[4]
    
    rec_type = "gmm_all"
    i_filter = False
    landmark_objects(dataset_path, dump_path, rec_type, db_path, res_dump, i_filter)

