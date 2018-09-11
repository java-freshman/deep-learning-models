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

gflags.DEFINE_string('results_dir',
                     'results',
                     'path of the result dir')
gflags.DEFINE_string('feat_folder',
                     'extract_feature',
                     'path of the feature folder')
gflags.DEFINE_string('retrieval_folder',
                     'similarity_retrieval',
                     'path of the feature folder')
gflags.DEFINE_string('img_vect',
                     'img_vect',
                     'embedding img vectors')
gflags.DEFINE_string('img_name',
                     'img_name',
                     'name for images')
FLAGS = gflags.FLAGS


def load_data(img_name_file, img_vect_file):
    """

    :param img_name_file:
    :param img_vect_file:
    :return:
    """
    img_vect_list = np.load(img_vect_file)
    img_name_list = pickle.load(open(img_name_file, 'rb'))
    img_name_list = [int(x.split('_')[0]) for x in img_name_list]

    return img_vect_list, img_name_list


def main(argv):
    FLAGS(argv)

    img_vect_path = os.path.join(FLAGS.results_dir, FLAGS.feat_folder,
                                 FLAGS.img_vect)

    img_name_path = os.path.join(FLAGS.results_dir, FLAGS.feat_folder,
                                 FLAGS.img_name)

    file_list = os.listdir(img_name_path)

    for file in file_list:

        img_name_file = os.path.join(img_name_path, file)
        img_vect_file = os.path.join(img_vect_path, file + '.npy')
        img_vect_list, img_name_list = load_data(img_name_file, img_vect_file)

        start = time.time()

        if len(img_name_list) <= 50:
            continue

        img_vec_mat = list()
        for prd_id in img_name_list:
            idx = img_name_list.index(prd_id)
            vec = img_vect_list[idx]
            img_vec_mat.append(vec)
        img_vec_mat = np.asarray(img_vec_mat)
        print("{}: transform {} img vectors into matrix in "
              "{:.4f} secs".format(file, len(img_name_list), time.time() - start))

        start = time.time()
        d = img_vec_mat @ img_vec_mat.T
        norm = img_vec_mat.T * img_vec_mat.T
        norm = norm.sum(0, keepdims=True) ** .5
        M = d / norm / norm.T
        print("{}: compute similarity matrix within {:.3f} "
              "secs".format(file, time.time() - start))

        start = time.time()
        prdid_similar_dict = dict()
        for i in range(M.shape[0]):
            prdid_i = img_name_list[i]
            tmp_dict = dict()
            for j in range(M.shape[1]):
                if i == j:
                    continue
                prdid_j = img_name_list[j]
                tmp_dict[prdid_j] = M[i, j]
            prdid_similar_dict[prdid_i] = sorted(
                tmp_dict.items(), key=lambda item: item[1])[-50:]
        print("{}: finish fetching prdid similar dict within"
              " {:.4f} secs\n".format(file, time.time() - start))

        similar_retrieval_path = os.path.join(
            FLAGS.results_dir, FLAGS.retrieval_folder)
        if not os.path.exists(similar_retrieval_path):
            os.makedirs(similar_retrieval_path)
        pickle.dump(prdid_similar_dict,
                    open(similar_retrieval_path + '/' +
                         file + '_' + 'prdid_similar_dict', 'wb'))


if __name__ == '__main__':
    main(sys.argv)
