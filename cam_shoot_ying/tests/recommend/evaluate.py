import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/cam_shoot_ying/')

import prediction.evaluation as evals

def process(dataset_path, dump_path, features_path, test_path):
    # basic
    evals.process_dataset(dataset_path, dump_path, features_path, test_path, rec_type='basic')

    # with time
    evals.process_dataset(dataset_path, dump_path, features_path, test_path, rec_type='time')

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print "Usage : dataset_path dump_path features_path test_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    features_path = sys.argv[3]
    test_path = sys.argv[4]
    
    process(dataset_path, dump_path, features_path, test_path)

