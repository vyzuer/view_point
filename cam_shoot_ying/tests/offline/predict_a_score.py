import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/cam_shoot_ying/')

import prediction.predict_score as predict

def process(model_path, dump_path):
    predict.process_dataset(model_path, dump_path)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : model_path dump_path"
        sys.exit(0)

    model_path = sys.argv[1]
    dump_path = sys.argv[2]
    
    process(model_path, dump_path)

