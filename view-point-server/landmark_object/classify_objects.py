import sys
import time
import os
import glob
import numpy as np
import shutil

from sklearn.externals import joblib

def dump_cluster_info(lmo_id, cl_dump, a_score, weather_info, geo_info, time_info, saliency_info):
    
    cl_dump_path = cl_dump + str(lmo_id)
    if not os.path.exists(cl_dump_path):
        os.makedirs(cl_dump_path)

    f_img_aesthetic_score = cl_dump_path + "/aesthetic.scores"
    f_weather_data = cl_dump_path + "/weather.info"
    f_geo_info = cl_dump_path + "/geo.info"
    f_time_info = cl_dump_path + "/time.info"
    f_saliency_info = cl_dump_path + "/saliency.info"

    fp_a_score = open(f_img_aesthetic_score, "a")
    fp_a_score.write("%.6f\n" % a_score)
    fp_a_score.close()

    fp_weather_data = open(f_weather_data, 'a')
    np.savetxt(fp_weather_data, np.atleast_2d(weather_info), fmt='%.6f')
    fp_weather_data.close()

    fp_geo_info = open(f_geo_info, 'a')
    np.savetxt(fp_geo_info, np.atleast_2d(geo_info), fmt='%.6f')
    fp_geo_info.close()

    fp_time_info = open(f_time_info, "a")
    fp_time_info.write("%s\n" % time_info)
    fp_time_info.close()

    fp_saliency_score = open(f_saliency_info, "a")
    fp_saliency_score.write("%.8f\n" % saliency_info)
    fp_saliency_score.close()


def classify_segments(image_dump_path, cl_dump, cluster_model, a_score, weather_info, geo_info, time_info, lmo_importance, img_id):

    f_feature_list = image_dump_path + "/feature.list"
    f_saliency_list = image_dump_path + "/saliency.list"

    feature_list = np.loadtxt(f_feature_list, ndmin=2)
    saliency_list = np.loadtxt(f_saliency_list, ndmin=1)
    n_segments = len(feature_list)

    for i in range(n_segments):
        lmo_id = cluster_model.predict(feature_list[i])

        dump_cluster_info(lmo_id[0], cl_dump, a_score, weather_info, geo_info, time_info, saliency_list[i])

        # modify lmo matrix
        lmo_importance[img_id][lmo_id[0]] += 1


def process_dataset(model_path, dump_path):
    
    # load cluster model
    f_model_path = model_path + "cluster_model/cluster.pkl"
    cluster_model = joblib.load(f_model_path)
    seg_dumps = dump_path + "segment_dumps/"

    cl_dump = dump_path + "lm_objects/"
    if os.path.exists(cl_dump):
        shutil.rmtree(cl_dump)

    if not os.path.exists(cl_dump):
        os.makedirs(cl_dump)

    image_list = dump_path + "image.list"
    fp_image_list = open(image_list, 'r')

    f_img_aesthetic_score = dump_path + "aesthetic.scores"
    f_weather_data = dump_path + "weather.info"
    f_geo_info = dump_path + "geo.info"
    f_time_info = dump_path + "time.info"

    a_score = np.loadtxt(f_img_aesthetic_score)
    weather_info = np.loadtxt(f_weather_data)
    geo_info = np.loadtxt(f_geo_info)
    time_info = np.loadtxt(f_time_info, dtype='string')

    # matrix for lmo importance
    num_images = len(geo_info)
    num_lmo = len(cluster_model.cluster_centers_indices_)
    lmo_importance = np.zeros(shape=(num_images, num_lmo), dtype='int')

    i = 0
    # Browse through each image
    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        image_name = os.path.splitext(image_name)[0]
        image_dump_path = seg_dumps + image_name

        # timer = time.time()
        # print image_dump_path

        classify_segments(image_dump_path, cl_dump, cluster_model, a_score[i], weather_info[i], geo_info[i], time_info[i], lmo_importance, i)
        
        # print "Total run time = ", time.time() - timer

        i += 1


    # find lmo importance
    # dump bag of visual words here for future use
    img_codebook = dump_path + 'img_codebook.list'
    np.savetxt(img_codebook, lmo_importance, fmt='%d')

    # lmo_codebook = np.loadtxt(img_codebook, dtype='int')
    lmo_codebook = lmo_importance
    # invert term frequency for each visual word
    lmo_itf = np.zeros(num_lmo)
    lmo_sal = np.zeros(num_lmo)

    # max_df = np.max(lmo_codebook.astype(bool).sum(0))

    # 
    # for i in range(num_images):
    #     for j in range(num_lmo):
    #         lmo_itf[j] += lmo_codebook[i][j]

    lmo_count = lmo_codebook.astype(bool).sum(0)
    lmo_itf = np.sum(lmo_codebook, axis=0)

    for j in range(num_lmo):
        if lmo_itf[j] != 0:
            lmo_itf[j] = (1.0*lmo_count[j]*np.log(lmo_count[j]))/lmo_itf[j]

    max_itf = max(lmo_itf)
    # lmo_df = lmo_itf/num_images
    # lmo_df = lmo_itf/max_df
    lmo_df = lmo_itf/max_itf

    for i in range(num_lmo):
        cl_dump_path = cl_dump + str(i)

        f_img_aesthetic_score = cl_dump_path + "/aesthetic.scores"
        f_saliency_info = cl_dump_path + "/saliency.info"

        saliency_list = np.loadtxt(f_saliency_info)
        a_score_list = np.loadtxt(f_img_aesthetic_score)
        
        lmo_sal[i] = np.average(saliency_list*a_score_list)

    max_lmo_sal = max(lmo_sal)
    lmo_sal = lmo_sal/max_lmo_sal
        
    lmo_importance = model_path + 'lmo_importance.list'
    np.savetxt(lmo_importance, lmo_df, fmt='%.8f')

    lmo_saliency = model_path + 'lmo_saliency.list'
    np.savetxt(lmo_saliency, lmo_sal, fmt='%.8f')

    avg_imp = (lmo_sal + lmo_df)/2.0

    max_avg_imp = max(avg_imp)
    avg_imp = avg_imp/max_avg_imp
        
    imp_file = model_path + 'importance.list'
    np.savetxt(imp_file, avg_imp, fmt='%.8f')

