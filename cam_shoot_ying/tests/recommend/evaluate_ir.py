import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python')

import prediction.evaluation_ir as evals

def landmark_objects(dataset_path, dump_path, db_path, res_dump, cuttoff):
    # create a sample test set for evaluation
    # only one time run, comment after that
    evals.create_testset(dataset_path, dump_path, db_path, res_dump)

    # evaluate on the generated testset
    evals.process_dataset(dataset_path, dump_path, db_path, res_dump, cuttoff)

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print "Usage : dataset_path dump_path db_path res_dump"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    db_path = sys.argv[3]
    res_dump = sys.argv[4]
    
    cuttoff = 0.68
    num_iter = 1
    for i in range(num_iter):
        landmark_objects(dataset_path, dump_path, db_path, res_dump, cuttoff)
        cuttoff += 0.01

