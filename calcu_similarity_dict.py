"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/9/9
"""
import os
import sys
import time
import pickle

import numpy as np

import gflags

gflags.DEFINE_string('feat_dir',
                     'extract_feature',
                     'path of the feature folder')
gflags.DEFINE_string('embedding_vec',
                     'embedding_vec',
                     'embedding vectors')
gflags.DEFINE_string('img_name',
                     'img_name',
                     'product id')
gflags.DEFINE_string('new_dcd',
                     'B43130301',
                     'version of the extracted feature')
FLAGS = gflags.FLAGS

def load_data():
    newdcd_cate_prdid = pickle.load(open('B43_newdcd_cate_prdid', 'rb'))

    prd_img_vec_file = os.path.join(
            FLAGS.feat_dir,
            FLAGS.embedding_vec,
            FLAGS.new_dcd+".npy")
    prdid_name_file = os.path.join(
            FLAGS.feat_dir,
            FLAGS.img_name,
            FLAGS.new_dcd)

    prd_img_vec = np.load(prd_img_vec_file)

    prdid_list = pickle.load(open(prdid_name_file, 'rb'))
    prdid_list = [int(x.split('_')[0]) for x in prdid_list]

    return newdcd_cate_prdid, prd_img_vec, prdid_list


def main(argv):
    FLAGS(argv)

    newdcd_cate_prdid, prd_img_vec, prdid_list = load_data()
    for cate in newdcd_cate_prdid[FLAGS.new_dcd].keys():
        start = time.time()

        tmp_prdid_list = list(set(prdid_list).intersection(
                newdcd_cate_prdid[FLAGS.new_dcd][cate]))
        if len(tmp_prdid_list)<= 50:
            continue

        embedding_img_vec_mat = list()
        for prd_id in  tmp_prdid_list:
            idx = prdid_list.index(prd_id)
            vec = prd_img_vec[idx]
            embedding_img_vec_mat.append(vec)
        embedding_img_vec_mat = np.asarray(embedding_img_vec_mat)
        print("category {}: transform {} img vectors into matrix "
              "in {} secs".format(cate, len(tmp_prdid_list), time.time()-start))

        start = time.time()
        d = embedding_img_vec_mat@embedding_img_vec_mat.T
        norm = embedding_img_vec_mat.T*embedding_img_vec_mat.T
        norm = norm.sum(0,keepdims=True) ** .5
        M = d/norm/norm.T
        print("category {}: compute similarity matrix within {} "
              "secs".format(cate, time.time() - start))

        start = time.time()
        prdid_similar_dict = dict()
        for i in range(M.shape[0]):
            prdid_i = tmp_prdid_list[i]
            tmp_dict = dict()
            for j in range(M.shape[1]):
                if i == j:
                    continue
                prdid_j = tmp_prdid_list[j]
                tmp_dict[prdid_j] = M[i,j]
            prdid_similar_dict[prdid_i] = sorted(
                    tmp_dict.items(), key=lambda item:item[1])[-50:]
        print("category {}: finish fetching prdid similar dict within"
              " {} secs".format(cate, time.time() - start))
        print("\n")
        pickle.dump(
                prdid_similar_dict,
                open('img_similarity_retrieval/mobilenet_crop/'+
                     FLAGS.new_dcd+'_'+str(cate)+'_'+'prdid_similar_dict','wb'))

if __name__ == '__main__':
    main(sys.argv)
