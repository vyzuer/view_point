from skimage import io
import glob
import os, sys
import numpy as np
import shutil
from sklearn.cross_validation import train_test_split

def create_testbed(db_path, dump_path):
    image_list = db_path + '/image.list'
    image_dir = db_path + "/ImageDB/"

    f_geo = db_path + "/geo.info"
    f_time = db_path + "/time.info"
    f_env = db_path + "/weather.info"
    f_aes = db_path + "/aesthetic.scores"

    img_list = np.loadtxt(image_list, dtype='string')
    geo_list = np.loadtxt(f_geo)
    time_list = np.loadtxt(f_time, dtype='string')
    env_list = np.loadtxt(f_env)
    a_score = np.loadtxt(f_aes)

    idx = np.arange(a_score.size)

    img_list_train, img_list_test, idx_train, idx_test, geo_list_train, geo_list_test, time_list_tain, time_list_test, env_list_train, env_list_test, a_score_train, a_score_test = train_test_split(img_list, idx, geo_list, time_list, env_list, a_score, test_size=0.05, random_state=42)

    image_list = dump_path + '/image.list'
    f_geo = dump_path + "/geo.info"
    f_time = dump_path + "/time.info"
    f_env = dump_path + "/weather.info"
    f_aes = dump_path + "/aesthetic.scores"
    f_idx = dump_path + "/idx.list"

    np.savetxt(image_list, img_list_test, fmt='%s')
    np.savetxt(f_geo, geo_list_test, fmt='%.8f')
    np.savetxt(f_time, time_list_test, fmt='%s')
    np.savetxt(f_env, env_list_test, fmt='%.8f')
    np.savetxt(f_aes, a_score_test, fmt='%.8f')
    np.savetxt(f_idx, idx_test, fmt='%d')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    db_dir = sys.argv[1]
    dump_path = sys.argv[2]

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    create_testbed(db_dir, dump_path)

