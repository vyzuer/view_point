import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/cam_shoot_ying/')

import landmark_object.popularity as pop

def process(dataset_path, dump_path):
    # compute ppopularity of each of the viewpoints
    # for 2d
    pop.process_dataset(dataset_path, dump_path)
    # for 3d
    pop.process_dataset(dataset_path, dump_path, gmm_3d=True)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process(dataset_path, dump_path)



