import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/img_retrieval_BM/')

import landmark_object.detection as detect

def landmark_objects(dataset_path, dump_path):
    detect.process(dataset_path, dump_path)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    landmark_objects(dataset_path, dump_path)

