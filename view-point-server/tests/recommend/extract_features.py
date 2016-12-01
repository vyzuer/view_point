import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python')

import common.extract_features as xtr

def process(dataset_path, dump_path):
    xtr.process_dataset(dataset_path, dump_path)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    
    process(dataset_path, dump_path)

