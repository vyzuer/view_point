from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from sklearn.externals import joblib
from sklearn import preprocessing
import scipy
import shutil
import random

from common import salient_object_detection as obj_detection

img_features = None
img_codebook = None

score_cuttoff = 0.68

def euclidean_distance(id_src, id_dst):
    src_fv = img_features[id_src]
    dst_fv = img_features[id_dst]

    src_cb = img_codebook[id_src]
    dst_cb = img_codebook[id_dst]

    fv_dist = np.linalg.norm(src_fv-dst_fv)
    cb_dist = np.linalg.norm(src_cb-dst_cb)

    # print fv_dist, cb_dist

    dist = fv_dist + cb_dist
    # dist = cb_dist
    dist = fv_dist

    return dist


def retrieve_images(qid, idx_list):
    # iterate over all the images in the dataset
    # to find similar images

    num_imgs = idx_list.size

    similarity_score = np.zeros(num_imgs)

    for i in range(num_imgs):
        similarity_score[i] = euclidean_distance(qid, idx_list[i])

    similar_images = np.argsort(similarity_score)

    scores = np.sort(similarity_score)

    # print scores

    return similar_images, scores


def find_images_in_geopixel(src_lat_idx, src_lon_idx, lat_idx, lon_idx, db_path):

    idx_list = []

    num_images = len(lat_idx)

    for i in range(num_images):
        if src_lat_idx == lat_idx[i] and src_lon_idx == lon_idx[i]:
            idx_list.append(i)

    return np.asarray(idx_list)


def images_similar(id_src, id_dst):
    src_fv = img_features[id_src]
    dst_fv = img_features[id_dst]

    src_cb = img_codebook[id_src]
    dst_cb = img_codebook[id_dst]

    fv_dist = np.linalg.norm(src_fv-dst_fv)
    cb_dist = np.linalg.norm(src_cb-dst_cb)

    # print fv_dist, cb_dist

    similar = False
    if fv_dist < 3 or cb_dist < 7:
        similar = True

    return similar

def find_similar_images(idx, img_list):

    idx_list = []

    for img in img_list:
        if images_similar(idx, img):
            idx_list.append(img)

    return np.asarray(idx_list)

def get_src_image_list(idx, lat_idx, lon_idx, db_path):

    src_lat_idx = lat_idx[idx]
    src_lon_idx = lon_idx[idx]

    src_img_list = find_images_in_geopixel(src_lat_idx, src_lon_idx, lat_idx, lon_idx, db_path)

    # print src_img_list.size

    assert src_img_list.size != 0

    return src_img_list[:30]

def find_src_image_list(idx, lat_idx, lon_idx, db_path):

    src_lat_idx = lat_idx[idx]
    src_lon_idx = lon_idx[idx]

    src_img_list = find_images_in_geopixel(src_lat_idx, src_lon_idx, lat_idx, lon_idx, db_path)

    # print src_img_list.size

    src_similar_img_list = find_similar_images(idx, src_img_list)

    # print src_similar_img_list.size

    assert src_similar_img_list.size != 0

    return src_similar_img_list

def get_dst_image_list(reco_pos, lat_idx, lon_idx, db_path, g_shape, idx):
    p_lat, p_lon = np.unravel_index(reco_pos, g_shape)
    
    # find images from this geo pixel
    dst_img_list = find_images_in_geopixel(p_lat, p_lon, lat_idx, lon_idx, db_path)

    # print dst_img_list.size

    return dst_img_list


def find_dst_image_list(reco_pos, lat_idx, lon_idx, db_path, g_shape, idx):
    p_lat, p_lon = np.unravel_index(reco_pos, g_shape)
    # find images from this geo pixel
    dst_img_list = find_images_in_geopixel(p_lat, p_lon, lat_idx, lon_idx, db_path)

    # print dst_img_list.size

    dst_similar_img_list = find_similar_images(idx, dst_img_list)
    # assert dst_img_list.size != 0

    # print dst_similar_img_list.size

    return dst_similar_img_list

def find_geo_pixel_score(img_idx_list, db_path):
    f_ascore = db_path + '/aesthetic.scores'

    a_score = np.loadtxt(f_ascore)

    score = 0.0
    for img_idx in img_idx_list:
        score += a_score[img_idx]

    return score/a_score.size

def dump_images_with_list(img_list_idx, db_path, img_list, dump_dir, dump_name):

    img_dump_dir = dump_dir + '/' + dump_name
    if not os.path.exists(img_dump_dir):
        os.makedirs(img_dump_dir)

    f_img_idx_list = dump_dir + '/' + dump_name + '_idx.list'

    np.savetxt(f_img_idx_list, img_list_idx, fmt='%d')

    f_img_list = dump_dir + '/' + dump_name + '.list'
    fp = open(f_img_list, 'w')

    for i in img_list_idx:
        img_name = img_list[i]
        fp.write("%s\n" %(img_name))

        src = db_path + '/ImageDB/' + img_name
        dst = img_dump_dir + '/' + img_name

        os.symlink(src, dst)

    fp.close()


def dump_images(img_list_idx, db_path, img_list, dump_dir, dump_name):

    dump_dir = dump_dir + '/' + dump_name
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    for i in img_list_idx:
        img_name = img_list[i]

        src = db_path + '/ImageDB/' + img_name
        dst = dump_dir + '/' + img_name

        os.symlink(src, dst)


def compute_prec(rel_list, p):
    avg_p = 0.0

    for i in range(0, p):
        avg_p += rel_list[i]

    return avg_p/p

def compute_ndcg(rel_list, p):
    ndcg = 0.0

    for i in range(0, p):
        ndcg += (np.power(2, rel_list[i]) -1)/np.log2(i+2)

    idcg = 0.0

    sorted_rel_list = np.sort(rel_list)[::-1]

    for i in range(0, p):
        idcg += (np.power(2, sorted_rel_list[i]) -1)/np.log2(i+2)

    if idcg != 0:
        ndcg /= idcg

    return ndcg

def evaluate_rec(idx, lat_idx, lon_idx, db_path, dump_path, db_src, res_dump, rec_type, rec_geo_pixels, g_shape, top_k=1, i_filter=False):

    f_image_list = db_path + '/image.list'
    img_list = np.loadtxt(f_image_list, dtype='string')

    # find all the images in the source geo-pixel with similar view
    src_img_idx_list = find_src_image_list(idx, lat_idx, lon_idx, db_path)

    # dump the src images
    input_img_name = img_list[idx]

    img_name = os.path.splitext(input_img_name)[0]

    dump_dir = res_dump + '/reco_dump/'
    if i_filter == True:
        dump_dir = res_dump + '/reco_dump_1/'

    dump_dir = dump_dir + rec_type + '/' + img_name 

    # if os.path.exists(dump_dir):
    #    shutil.rmtree(dump_dir)

    # os.makedirs(dump_dir)

    src = db_src + '/ImageDB/' + input_img_name
    dst = dump_dir + '/' + input_img_name

    # os.symlink(src, dst)

    # dump_images(src_img_idx_list, db_src, img_list, dump_dir, 'src')
    
    src_score = find_geo_pixel_score(src_img_idx_list, db_path)

    dst_score =  np.zeros(top_k)
    dst_rel =  np.zeros(top_k)

    dst_img_idx_list = find_dst_image_list(rec_geo_pixels[0], lat_idx, lon_idx, db_path, g_shape, idx)
    # dump_images(dst_img_idx_list, db_src, img_list, dump_dir, 'dst')
    for i in range(top_k):
        dst_img_idx_list = find_dst_image_list(rec_geo_pixels[i], lat_idx, lon_idx, db_path, g_shape, idx)
        dst_rel[i] = find_geo_pixel_score(dst_img_idx_list, db_path)

        if src_score < dst_rel[i]:
            dst_score[i] = 1

    p1 = compute_prec(dst_score, 1)
    p2 = compute_prec(dst_score, 2)
    p5 = compute_prec(dst_score, 5)

    r1 = compute_ndcg(dst_rel, 1)
    r2 = compute_ndcg(dst_rel, 2)
    r5 = compute_ndcg(dst_rel, 5)

    return p1, p2, p5, r1, r2, r5

def set_global_vars(db_path):
    global img_features
    global img_codebook

    f_img_features = db_path + '/features.list'
    f_img_codebook = db_path + '/img_codebook.list'

    img_features = np.loadtxt(f_img_features)
    img_codebook = np.loadtxt(f_img_codebook)

    # normalize data
    img_features = preprocessing.normalize(img_features, norm='l2')
    img_codebook = preprocessing.normalize(img_codebook, norm='l2')

def find_geo_density(db_path):

    assert os.path.exists(db_path)

    # make a matrix map of this location
    geo_minmax_list = db_path + "geo_minmax.list"
    geo_minmax = np.loadtxt(geo_minmax_list)

    g_xmin, g_xmax = geo_minmax[0], geo_minmax[1]
    g_ymin, g_ymax = geo_minmax[2], geo_minmax[3]
    g_xstep, g_ystep = geo_minmax[4], geo_minmax[5]
    num_xstep, num_ystep = int((g_xmax - g_xmin)/g_xstep), int((g_ymax - g_ymin)/g_ystep)

    f_geo = db_path + "/geo.info"
    geo_list = np.loadtxt(f_geo)

    density = 1.0*geo_list.shape[0]/(num_xstep*num_ystep)

    print density

    density_file = db_path + '/density.info'
    fp = open(density_file, 'w')
    fp.write('%f' %(density))
    fp.close()


def create_db(idx, lat_idx, lon_idx, db_path, dump_path, db_src, res_dump, rec_geo_pixels, g_shape):

    img_count = 0

    f_image_list = db_path + '/image.list'
    img_list = np.loadtxt(f_image_list, dtype='string')

    # find all the images in the source geo-pixel 
    src_img_idx_list = get_src_image_list(idx, lat_idx, lon_idx, db_path)
    
    img_count += src_img_idx_list.size

    # dump the src images
    input_img_name = img_list[idx]

    img_name = os.path.splitext(input_img_name)[0]

    dump_dir = dump_path + '/ImageDB/' + img_name

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    src = db_src + '/ImageDB/' + input_img_name
    dst = dump_dir + '/' + input_img_name

    os.symlink(src, dst)

    dump_images_with_list(src_img_idx_list, db_src, img_list, dump_dir, 'src')
    
    dst_img_idx_list = get_dst_image_list(rec_geo_pixels[0], lat_idx, lon_idx, db_path, g_shape, idx)
    dump_images_with_list(dst_img_idx_list, db_src, img_list, dump_dir, 'dst')

    img_count += dst_img_idx_list.size

    return input_img_name, img_count

def create_testset(db_path, dump_path, db_src, res_dump, db_size=10):
    """
    db_path - all features are stored here
    dump_path - dump new dataset here
    db_src - all images are stored here
    res_dump - recommendation results are stored here
        """

    img_count = 0

    # if the testset is already present just skip thi step
    b_valid_file = dump_path + '.valid'
    if os.path.exists(b_valid_file):
        print "\nTestset already present, clean valid file to recreate it.\n"
        return

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    
    f_db_idx_list = dump_path + '/idx.list'
    f_image_list = dump_path + '/image.list'

    fp_idx = open(f_db_idx_list, 'w')
    fp_img_list = open(f_image_list, 'w')

    f_id_list = res_dump + '/idx.list'
    idx_list = np.loadtxt(f_id_list, dtype='int')

    num_samples = idx_list.size

    res_file = res_dump + '/gmm_basic.results'
    rec_geo_pixels = np.loadtxt(res_file, dtype='int')

    random_samples_id = random.sample(range(num_samples), db_size)

    geo_minmax_list = db_path + "geo_minmax.list"
    geo_minmax = np.loadtxt(geo_minmax_list)

    g_xmin, g_xmax = geo_minmax[0], geo_minmax[1]
    g_ymin, g_ymax = geo_minmax[2], geo_minmax[3]
    g_xstep, g_ystep = geo_minmax[4], geo_minmax[5]
    num_xstep, num_ystep = int((g_xmax - g_xmin)/g_xstep), int((g_ymax - g_ymin)/g_ystep)

    f_geo = db_path + "/geo.info"
    geo_list = np.loadtxt(f_geo)

    hist, e_lat, e_lon = np.histogram2d(geo_list[:,0], geo_list[:,1], bins=[num_xstep, num_ystep])
    lat_idx = np.digitize(geo_list[:,0], e_lat)-1 # -1 to offset the index for arrays
    lon_idx = np.digitize(geo_list[:,1], e_lon)-1

    for i in range(db_size):
        idx = random_samples_id[i]
        # print idx
        img_name, cnt = create_db(idx, lat_idx, lon_idx, db_path, dump_path, db_src, res_dump, rec_geo_pixels[idx], hist.shape)
        img_count += cnt
        fp_idx.write("%d\n" % (idx))
        fp_img_list.write("%s\n" %(img_name))

    print 'Total Images : ', img_count

    fp_idx.close()
    fp_img_list.close()

    np.savetxt(b_valid_file, [img_count], fmt='%d')


def eval_ir(img_name, idx, dump_dir, top_k = 10):
    prec = 0.0
    rec = 0.0

    # evaluate src
    f_src_idx = dump_dir + '/src_idx.list'
    f_gt_src = dump_dir + '/gt_src.list'
    src_idx_list = np.loadtxt(f_src_idx, dtype='int')
    src_gt = np.loadtxt(f_gt_src, dtype='int')

    num_rel_imgs = np.sum(src_gt)

    img_list, scores = retrieve_images(idx, src_idx_list)
    p = 0.0

    # print img_list
    num_images = img_list.size

    for j in range(num_images):
        # break if images imilarity is too much
        if scores[j] > score_cuttoff:
            break

        if 1 == src_gt[img_list[j]]:
            p += 1.0

    prec += p/j
    rec += p/num_rel_imgs

    # print img_name
    # print 'src\t', prec, num_rel_imgs

    # evaluate dst
    f_dst_idx = dump_dir + '/dst_idx.list'
    f_gt_dst = dump_dir + '/gt_dst.list'
    dst_idx_list = np.loadtxt(f_dst_idx, dtype='int')
    dst_gt = np.loadtxt(f_gt_dst, dtype='int')

    num_rel_imgs = np.sum(dst_gt)

    img_list, scores = retrieve_images(idx, dst_idx_list)
    p = 0.0

    # print img_list
    num_images = img_list.size

    for j in range(num_images):
        # break if images imilarity is too much
        if scores[j] > score_cuttoff:
            break

        if 1 == dst_gt[img_list[j]]:
            p += 1.0

    prec += p/j
    rec += p/num_rel_imgs

    # print 'dst\t', p/num_rel_imgs, num_rel_imgs, '\n'

    return prec/2, rec/2

def process_dataset(db_path, dump_path, db_src, res_dump, cuttoff = 0.68):

    assert os.path.exists(db_path)
    assert os.path.exists(dump_path)

    # load global features for future use
    set_global_vars(db_path)

    global score_cuttoff
    score_cuttoff = cuttoff

    result_file = dump_path + '/' + 'result.map'
    fp = open(result_file, 'w')

    f_idx_list = dump_path + 'idx.list'
    f_img_list = dump_path + 'image.list'

    idx_list = np.loadtxt(f_idx_list, dtype='int')
    img_list = np.loadtxt(f_img_list, dtype='string')
    
    num_query_samples = idx_list.size

    prec = 0.0
    rec = 0.0

    for i in range(num_query_samples):

        img_name = os.path.splitext(img_list[i])[0]
        idx = idx_list[i]
        
        dump_dir = dump_path + 'ImageDB/' + img_name

        p, r = eval_ir(img_name, idx, dump_dir)

        prec += p
        rec += r

    prec_score = prec/num_query_samples
    rec_score = rec/num_query_samples
    f1_score = 2*prec_score*rec_score/(prec_score + rec_score)

    print 'MAP = ', prec_score
    print 'Recall = ', rec_score
    print 'F1 = ', f1_score
    fp.write("MAP : %f" % (prec_score))
    fp.write("Recall : %f" % (rec_score))
    fp.write("Recall : %f" % (f1_score))

    fp.close()


