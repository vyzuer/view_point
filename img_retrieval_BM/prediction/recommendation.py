from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from sklearn.externals import joblib
from sklearn import preprocessing
import scipy
import shutil

from common import salient_object_detection as obj_detection

def find_recommendation(img_src, dump_path, vp_model_path, geo_info, rec_type="gmm_basic", gp_filter=False, time_info=None, weather_info=None):

    my_object = obj_detection.SalientObjectDetection(img_src, max_iter_slic=20)

    lm_objects_list, saliency_list = my_object.classify_objects(dump_path, vp_model_path, seg_dump=False)
    num_faces = my_object.num_of_faces()

    img_name = os.path.split(img_src)[1]
    timer = time.time()

    if rec_type == 'gmm_all':
        rec_type_list = ['gmm_basic', 'gmm_time', 'gmm_weather']
        for r_type in rec_type_list:
            solve_recsys(lm_objects_list, saliency_list, vp_model_path, dump_path, num_faces, rec_type=r_type,  gp_filter=gp_filter, time_info=time_info, weather_info=weather_info, dump_map=True, img_name=img_name)
    else:
        solve_recsys(lm_objects_list, saliency_list, vp_model_path, dump_path, num_faces, rec_type=rec_type, gp_filter=gp_filter, time_info=time_info, weather_info=weather_info, dump_map=True, img_name=img_name)

    print("recsys runtime : %f" %(time.time() - timer))

def solve_recsys(lm_objects_list, saliency_list, vp_model_path, dump_path, num_faces, rec_type, gp_filter, time_info=None, weather_info=None, dump_map=False, img_name="reco.png"):

    num_objects = len(lm_objects_list)

    # object importance
    imp_list_path = vp_model_path + "importance.list"
    imp_list = np.asarray(np.loadtxt(imp_list_path))

    geo_minmax_list = vp_model_path + "geo_minmax.list"
    geo_minmax = np.loadtxt(geo_minmax_list)

    g_xmin, g_xmax = geo_minmax[0], geo_minmax[1]
    g_ymin, g_ymax = geo_minmax[2], geo_minmax[3]
    g_xstep, g_ystep = geo_minmax[4], geo_minmax[5]
    num_xstep, num_ystep = int((g_xmax - g_xmin)/g_xstep), int((g_ymax - g_ymin)/g_ystep)
    # print g_xmin, g_xmax, g_ymin, g_ymax, num_xstep, num_ystep

    geo_min_pos = np.array([g_xmin, g_ymin])
    geo_max_pos = np.array([g_xmax, g_ymax])

    if rec_type == "gmm_time":
        assert time_info is not None
        t_array = time_info.split(':')
        t_hrs = int(t_array[0]) + int(t_array[1])/60.0

        geo_min_pos = np.append(geo_min_pos, [0])
        geo_max_pos = np.append(geo_max_pos, [t_hrs])

    elif rec_type == "gmm_weather":
        assert time_info is not None
        assert weather_info is not None

        t_array = time_info.split(':')
        t_hrs = int(t_array[0]) + int(t_array[1])/60.0

        # minmax_info = vp_model_path + 'weather.minmax'
        # w_minmax_info = np.loadtxt(minmax_info)
        # w_minmax_info = scipy.delete(w_minmax_info, 7, 1)[:, :-3]
        geo_min_pos = np.concatenate([geo_min_pos, [0], weather_info])
        geo_max_pos = np.concatenate([geo_max_pos, [t_hrs], weather_info])

    heat_map = []
    # min_max_scaler = preprocessing.MinMaxScaler()

    for i in range(num_objects):
        # print lm_objects_list[i][0]

        # timer = time.time()

        obj_id = lm_objects_list[i][0]

        # load gmm models
        model_base_path = vp_model_path + "/gmm_models/" + str(obj_id) + "/" + rec_type
        gmm_model_path = model_base_path + "/model/gmm.pkl"
        scaler_model_path = model_base_path + "/scaler/scaler.pkl"
        mm_scaler_model_path = model_base_path + "/scaler/mm_scaler.pkl"

        f = np.zeros(shape=(num_xstep, num_ystep))
        if os.path.exists(gmm_model_path):

            gmm_model = joblib.load(gmm_model_path)
            scaler_model = joblib.load(scaler_model_path)
            mm_scaler_model = joblib.load(mm_scaler_model_path)

            scaled_min = scaler_model.transform(geo_min_pos)
            scaled_max = scaler_model.transform(geo_max_pos)
            
            xmin, xmax = scaled_min[0], scaled_max[0]
            ymin, ymax = scaled_min[1], scaled_max[1]

            x, y = np.mgrid[xmin:xmax:num_xstep*1j, ymin:ymax:num_ystep*1j]
            positions = np.vstack([x.ravel(), y.ravel()]).T

            if rec_type == "gmm_time":
                t_info = np.array([[scaled_max[2]],]*num_xstep*num_ystep)
                positions = np.concatenate([positions, t_info], axis=1)

            elif rec_type == "gmm_weather":
                sw_info = np.array([scaled_max[2:],]*num_xstep*num_ystep)
                positions = np.concatenate([positions, sw_info], axis=1)


            prob_score, response = gmm_model.score_samples(positions)
            prob_score = np.exp(prob_score)
            prob_score = mm_scaler_model.transform(prob_score)
            f = np.reshape(prob_score, x.shape)
    
            f_geo_pixel_map_path = vp_model_path + "/gmm_models/" + str(obj_id) + "/geo_pixel_map/geo_pixel.map"
            assert os.path.isfile(f_geo_pixel_map_path)
            
            geo_pixel_map = np.loadtxt(f_geo_pixel_map_path)
            f = f*geo_pixel_map

            if gp_filter == True:
                f_geo_pixel_map_path = vp_model_path + "/gmm_models/" + str(obj_id) + "/geo_pixel_map/geo_pixel_i.map"
                assert os.path.isfile(f_geo_pixel_map_path)
                
                geo_pixel_map = np.loadtxt(f_geo_pixel_map_path)
                # f = f*geo_pixel_map
                f = f + geo_pixel_map


        # f = min_max_scaler.fit_transform(f)

        heat_map.append(f)

        # adjust the importance of object
        imp_list[obj_id] = (saliency_list[i] + imp_list[obj_id])/2

        # print("each object runtime : %f" %(time.time() - timer))

    # consider human faces
    if num_faces > 0:
        # load gmm models
        model_base_path = vp_model_path + "/gmm_models/" + "/human_obj/" + rec_type
        gmm_model_path = model_base_path + "/model/gmm.pkl"
        scaler_model_path = model_base_path + "/scaler/scaler.pkl"
        mm_scaler_model_path = model_base_path + "/scaler/mm_scaler.pkl"

        gmm_model = joblib.load(gmm_model_path)
        scaler_model = joblib.load(scaler_model_path)
        mm_scaler_model = joblib.load(mm_scaler_model_path)

        if rec_type == "gmm_weather":
            t_hrs = geo_max_pos[2]

            geo_min_pos = np.concatenate([[g_xmin, g_ymin, 0, num_faces], weather_info])
            geo_max_pos = np.concatenate([[g_xmax, g_ymax, t_hrs, num_faces], weather_info])

        scaled_min = scaler_model.transform(geo_min_pos)
        scaled_max = scaler_model.transform(geo_max_pos)

        xmin, xmax = scaled_min[0], scaled_max[0]
        ymin, ymax = scaled_min[1], scaled_max[1]

        x, y = np.mgrid[xmin:xmax:num_xstep*1j, ymin:ymax:num_ystep*1j]
        positions = np.vstack([x.ravel(), y.ravel()]).T

        if rec_type == "gmm_time":
            t_info = np.array([[scaled_max[2]],]*num_xstep*num_ystep)
            positions = np.concatenate([positions, t_info], axis=1)

        elif rec_type == "gmm_weather":
            sw_info = np.array([scaled_max[2:],]*num_xstep*num_ystep)
            positions = np.concatenate([positions, sw_info], axis=1)


        prob_score, response = gmm_model.score_samples(positions)
        prob_score = np.exp(prob_score)
        prob_score = mm_scaler_model.transform(prob_score)
        f = np.reshape(prob_score, x.shape)

        f_geo_pixel_map_path = vp_model_path + "/gmm_models/human_obj/geo_pixel_map/geo_pixel.map"
        assert os.path.isfile(f_geo_pixel_map_path)
        
        geo_pixel_map = np.loadtxt(f_geo_pixel_map_path)
        f = f*geo_pixel_map

        if gp_filter == True:
            f_geo_pixel_map_path = vp_model_path + "/gmm_models/human_obj/geo_pixel_map/geo_pixel_i.map"
            assert os.path.isfile(f_geo_pixel_map_path)
            
            geo_pixel_map = np.loadtxt(f_geo_pixel_map_path)
            #  f = f*geo_pixel_map
            f = f + geo_pixel_map


        # f = min_max_scaler.fit_transform(f)

        heat_map.append(f)

    # timer = time.time()
    heat_map = np.asarray(heat_map)
    rec_heat_map = np.zeros(shape=(num_xstep, num_ystep))
    for i in range(num_objects):
        obj_id = lm_objects_list[i][0]
        rec_heat_map += imp_list[obj_id]*heat_map[i]
        # rec_heat_map += heat_map[i]

    # add human object 
    if num_faces > 0:
        rec_heat_map += heat_map[num_objects]

    # rec_heat_map = min_max_scaler.fit_transform(rec_heat_map)

    # print("rec runtime : %f" %(time.time() - timer))

    if dump_map == True:
        map_plot_path = dump_path + "/reco_maps/" + rec_type
        if gp_filter == True:
            map_plot_path = dump_path + '/reco_maps_1/' + rec_type

        if not os.path.exists(map_plot_path):
            os.makedirs(map_plot_path)

        plot_dump_path = map_plot_path + '/' + img_name

        # read the map for plotting
        # map_image_src = vp_model_path + "/map.png"
        # location_map = plt.imread(map_image_src)

        # plt.xlim(ymin, ymax)
        # plt.ylim(xmin, xmax)
        # plt.imshow(np.rot90(rec_heat_map.T), cmap=plt.cm.gist_earth_r, extent=[ymin, ymax, xmin, xmax])
        # plt.imshow(location_map)
        plt.matshow(np.rot90(rec_heat_map.T), cmap=plt.cm.gist_earth_r)
        plt.axis('off')
        plt.savefig(plot_dump_path, dpi=50)
        plt.close()

    # find the top five locations for recommendation
    num_reco = 5
    geo_rec = np.zeros(num_reco)
    for i in range(num_reco):
        idx = rec_heat_map.argmax()
        pos = np.unravel_index(idx, rec_heat_map.shape)
        rec_heat_map[pos] = 0
        geo_rec[i] = idx

    reco_results = dump_path + '/' + rec_type + '.results'

    if gp_filter == True:
        reco_results = dump_path + '/' + rec_type + '_1.results'

    fp = open(reco_results, 'a')
    np.savetxt(fp, geo_rec.reshape((1,-1)), fmt='%d')
    fp.close()


def get_reduced_data(w_info, model_path):
    pca_dump = model_path + '/w_pca_model/pca.pkl'
    scaler_dump = model_path + '/w_pca_model/scaler.pkl'

    pca = joblib.load(pca_dump)
    scaler = joblib.load(scaler_dump)

    w_info = scipy.delete(w_info, 7, 1)[:, :-3]
    w_info = scaler.transform(w_info)
    w_info = pca.transform(w_info)

    return w_info


def process_dataset(db_path, model_path, dump_path, rec_type="gmm_basic", gp_filter=False):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    if rec_type == 'gmm_all':
        rec_type_list = ['gmm_basic', 'gmm_time', 'gmm_weather']
        for r_type in rec_type_list:
            reco_results = dump_path + '/' + r_type + '.results'
            if gp_filter == True:
                reco_results = dump_path + '/' + r_type + '_1.results'
            try:
                os.remove(reco_results)
            except OSError:
                pass
    else:
        reco_results = dump_path + '/' + rec_type + '.results'
        if gp_filter == True:
            reco_results = dump_path + '/' + rec_type + '_1.results'
        try:
            os.remove(reco_results)
        except OSError:
            pass


    image_list = db_path + '/image.list'
    image_dir = db_path + "/ImageDB/"
    fp_image_list = open(image_list, 'r')

    f_geo = db_path + "/geo.info"
    f_time = db_path + "/time.info"
    f_env = db_path + "/weather.info"
    f_aes = db_path + "/aesthetic.scores"

    f_idx_src = db_path + '/idx.list'
    f_idx_dst = dump_path + '/idx.list'
    shutil.copy2(f_idx_src, f_idx_dst)

    geo_list = np.loadtxt(f_geo)
    time_list = np.loadtxt(f_time, dtype='string')
    env_list = np.loadtxt(f_env)
    a_score = np.loadtxt(f_aes)

    # env_list = scipy.delete(env_list, 7, 1)[:, :-3]
    env_list = get_reduced_data(env_list, model_path)

    i = 0

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\r\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        find_recommendation(infile, dump_path, model_path, geo_list[i], rec_type=rec_type, gp_filter=gp_filter, time_info=time_list[i], weather_info=env_list[i])
        
        print "================================="
        print "Total run time = ", time.time() - timer
        print "=================================\n"

        i += 1

    fp_image_list.close()

