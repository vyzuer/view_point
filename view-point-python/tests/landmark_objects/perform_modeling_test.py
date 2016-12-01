import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python')

import landmark_object.classify_objects as cl_obj
import landmark_object.gmm_modeling as gmm_model

def process(cluster_model_path, dump_path):
    # preprocess
    # cl_obj.process_dataset(cluster_model_path, dump_path)

    # perform modeling
    ext = "gmm_time"
    gmm_model.process_context(cluster_model_path, dump_path, ext, model_type='time')

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : cluster_model dump_path"
        sys.exit(0)

    cluster_model_path = sys.argv[1]
    dump_path = sys.argv[2]

    process(cluster_model_path, dump_path)

