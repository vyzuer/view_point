from skimage import io
import glob
import os, sys
import numpy as np
import shutil
import matplotlib.pyplot as plt

def plot_hist(w_dir, dump_dir, file_list_0, file_list_1):

    plot_name = dump_dir + "/a_score_hist.png"

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    # traverse all the datasets
    i = 1
    plt.figure(figsize=(48,6))
    for location in glob.glob(os.path.join(w_dir, '*')):
        # print location
        file_0 = location + file_list_0
        file_1 = location + file_list_1

        x = np.loadtxt(file_0)
        ax = plt.subplot(2, 12, i)
        plt.hist(x, bins=500)
        plt.axis("off")
        # ax.set_ylim([0, 100])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        x = np.loadtxt(file_1)
        ax = plt.subplot(2, 12, i+12)
        plt.hist(x, bins=500)
        # plt.axis("off")
        # ax.set_ylim([0, 100])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        i += 1

    plt.savefig(plot_name)
    plt.close

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage : dataset_path"
        sys.exit(0)

    w_dir = sys.argv[1]
    dump_dir = "/home/vyzuer/"

    file_list_0 = '/aesthetic.scores.0'
    file_list_1 = '/aesthetic.scores.1'

    plot_hist(w_dir, dump_dir, file_list_0, file_list_1)

