"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/15
"""
import os
import sys
import time

import gflags
import numpy as np

# import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from models.vgg16 import VGG16

gflags.DEFINE_string('img_dir',
                     '/home/wutenghu/git_wutenghu/gs_images/pic_43',
                     'path of the image folder')
gflags.DEFINE_string('feat_dir',
                     'extract_feature',
                     'path of the feature folder')
gflags.DEFINE_string('embedding_vec',
                     'embedding_vec',
                     'embedding vectors')
gflags.DEFINE_string('img_name',
                     'img_name',
                     'product id')
gflags.DEFINE_string('extract_ver',
                     '20180817',
                     'version of the extracted feature')
FLAGS = gflags.FLAGS

def load_model(model_name='vgg16'):
    if model_name == 'vgg16':
        model = VGG16(include_top=True, weights='imagenet')
    elif model_name == 'vgg19':
        model = None
    return model

def main(argv):
    FLAGS(argv)

    # load-in model
    model = load_model(model_name='vgg16')

    img_name_list = os.listdir(FLAGS.img_dir)

    embedding_vec_npy = list()
    img_name_npy = list()
    now = time.time()
    count = 0
    total_img_num = len(img_name_list)
    for img_name in img_name_list:

        count += 1
        if np.random.rand() > 1:
            continue

        if (count%500)==0:
            print('process finished {:.2f} percentage.'.format(
                    100*count/total_img_num))

        img_path = os.path.join(FLAGS.img_dir, img_name)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            img_name_npy.append(int(img_name.split('.')[0]))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pred = model.predict(x)
            embedding_vec_npy.append(pred[0])
        except Exception as e:
            print(img_name)
            pass

    embedding_vec_npy = np.asarray(embedding_vec_npy)
    img_name_npy = np.asarray(img_name_npy)

    embedding_vec_path = os.path.join(FLAGS.feat_dir, FLAGS.embedding_vec)
    img_name_path = os.path.join(FLAGS.feat_dir, FLAGS.img_name)

    np.save(
            embedding_vec_path+"/"+FLAGS.extract_ver,
            embedding_vec_npy)
    np.save(
            img_name_path+"/"+FLAGS.extract_ver,
            img_name_npy)

    print("process finished in {} seconds".format(time.time()-now))

if __name__ == '__main__':
    main(sys.argv)
