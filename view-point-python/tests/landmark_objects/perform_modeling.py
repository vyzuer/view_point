import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python')

import landmark_object.classify_objects as cl_obj
import landmark_object.gmm_modeling as gmm_model
import landmark_object.geo_pixel_map as gpmap

def process(cluster_model_path, dump_path, model_type):
    # preprocess
    cl_obj.process_dataset(cluster_model_path, dump_path)

    # perform modeling
    # model_type = "weather"
    ext = "gmm_" + model_type
    gmm_model.process_context(cluster_model_path, dump_path, ext, model_type=model_type)
    gmm_model.process_human_object(cluster_model_path, dump_path, ext, model_type=model_type)

def process_geo_pixel_map(cluster_model_path, dump_path):

    gpmap.process_lmo(cluster_model_path, dump_path, dump_map=True)

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage : cluster_model dump_path gmm_type"
        sys.exit(0)

    cluster_model_path = sys.argv[1]
    dump_path = sys.argv[2]
    gmm_type = sys.argv[3]

    process(cluster_model_path, dump_path, gmm_type)

    # dump the geo-pixel map for each landmark object
    process_geo_pixel_map(cluster_model_path, dump_path)



