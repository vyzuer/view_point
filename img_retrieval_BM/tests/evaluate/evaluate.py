import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/img_retrieval_BM')

import prediction.evaluation as evals

def landmark_objects(dataset_path, dump_path, qdump_path):
    evals.process_dataset(dataset_path, dump_path, qdump_path)

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage : dataset_path dump_path qdump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    qdump_path = sys.argv[3]
    
    landmark_objects(dataset_path, dump_path, qdump_path)

