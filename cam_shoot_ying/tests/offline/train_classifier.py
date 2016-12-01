import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/cam_shoot_ying/')

import classify.bin_classify_dump as clf

def process(dataset_path, dump_path):
    clf.learn_composition(dataset_path, dump_path, grid_search=True)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    
    process(dataset_path, dump_path)

