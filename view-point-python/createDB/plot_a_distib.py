from skimage import io
import glob
import os, sys
import numpy as np
import shutil
import matplotlib.pyplot as plt

def plot_hist(w_dir, file_list_0, file_list_1):
    plot_name_0 = w_dir + "a_score_hist_0.png"
    plot_name_1 = w_dir + "a_score_hist_1.png"

    # x = np.loadtxt(file_list_0)
    # plt.hist(x, bins=250)
    # plt.savefig(plot_name_0)
    # plt.clf()

    x = np.loadtxt(file_list_1)
    plt.hist(x, bins=250)
    plt.savefig(plot_name_1)
    plt.close

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage : dataset_path"
        sys.exit(0)

    w_dir = sys.argv[1]

    file_list_0 = w_dir + '/aesthetic.scores.0'
    file_list_1 = w_dir + '/aesthetic.scores'

    plot_hist(w_dir, file_list_0, file_list_1)

