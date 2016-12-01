import sys
import cv2
import os
# sys.path.append(os.path.abspath("../src/"))
import time
import glob
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

def process_dataset(model_path, dump_path):

    pred_list = dump_path + "/ying_ascore.list"

    model_dumps_path = model_path + "/models/"

    comp_model_path = model_dumps_path + "comp/"
    dump_scalar = comp_model_path + "scalar.pkl"
    dump_pca = comp_model_path + "pca.pkl"
    dump_svm = comp_model_path + "svm.pkl"

    regressor = joblib.load(dump_svm)
    scalar = joblib.load(dump_scalar)
    pca = joblib.load(dump_pca)

    feature_file = dump_path + 'features.list'
    X = np.loadtxt(feature_file)

    X = scalar.transform(X)
    X = preprocessing.normalize(X, norm='l2')
    X = pca.transform(X)

    pred = regressor.predict_proba(X)

    np.savetxt(pred_list, pred[:,1], fmt='%f')

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : model_path, dump_path"
        sys.exit(0)

    model_path = sys.argv[1] 
    dump_path = sys.argv[2]

    process_dataset(model_path, dump_path)

