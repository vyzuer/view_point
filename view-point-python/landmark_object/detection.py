import sys

import dump_features as dump_f
import cluster as cluster

def process(dataset_path, dump_path):
    # dump visual word features
    dump_f.process_dataset(dataset_path, dump_path)

    # dump visual word model
    cluster.process(dump_path)


def process_all(dataset_path, dump_path):
    # dump visual word features
    dump_f.process_dataset_all(dataset_path, dump_path)

