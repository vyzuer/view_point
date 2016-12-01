import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/cam_shoot_ying/')

import landmark_object.gmm_modeling as gmm_model

def process(dataset_path, dump_path):
    # perform modeling
    gmm_model.process_context(dataset_path, dump_path, gmm_3d=False)
    gmm_model.process_context(dataset_path, dump_path, gmm_3d=True)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process(dataset_path, dump_path)



