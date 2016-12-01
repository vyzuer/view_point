import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/img_retrieval_BM/')

import landmark_object.classify_objects as cl_obj
import common.extract_features as xtr

def process(dataset_path, cluster_model_path, dump_path):
    # preprocess
    cl_obj.process_dataset(cluster_model_path, dump_path)

    # other features
    xtr.process_dataset(dataset_path, dump_path)


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage : dataset_path cluster_model dump_path "
        sys.exit(0)

    dataset_path = sys.argv[1]
    cluster_model_path = sys.argv[2]
    dump_path = sys.argv[3]

    process(dataset_path, cluster_model_path, dump_path)

