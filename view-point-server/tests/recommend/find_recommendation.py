import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python')

import prediction.recommendation as recsys

def landmark_objects(dataset_path, model_path, dump_path, rec_type, gp_filter):
    recsys.process_dataset(dataset_path, model_path, dump_path, rec_type=rec_type, gp_filter=gp_filter)

if __name__ == '__main__':

    if len(sys.argv) != 6:
        print "Usage : dataset_path model_path dump_path rec_type"
        sys.exit(0)

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
    dump_path = sys.argv[3]
    rec_type = sys.argv[4]
    gp_filter_val = sys.argv[5]
    
    # rec_type = "gmm_all"
    gp_filter = False
    if gp_filter_val == 'True':
        gp_filter = True
    landmark_objects(dataset_path, model_path, dump_path, rec_type, gp_filter)

