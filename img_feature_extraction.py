"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/15
"""
import os
import sys
import time

import gflags
import numpy as np

import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from models.vgg16 import VGG16

gflags.DEFINE_string('img_dir',
                     '/home/wutenghu/git_wutenghu/neural_image_assessment/gs_images/pic_43_view_bt_5',
                     'path of the image folder')
gflags.DEFINE_string('feat_dir',
                     'extract_feature',
                     'path of the feature folder')
gflags.DEFINE_string('embedding_vec',
                     'embedding_vec',
                     'embedding vectors')
gflags.DEFINE_string('photo_pix',
                     'photo_pix',
                     'np type of the photos')
gflags.DEFINE_string('extract_ver',
                     '20180815',
                     'version of the extracted feature')
FLAGS = gflags.FLAGS

def resize_image(array, image_size):
    resize_tensor = tf.image.resize_images(
            array,
            (image_size, image_size),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session().as_default():
        result = resize_tensor.eval()
    return result

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
    photo_pix_npy = list()
    now = time.time()
    count = 0
    total_img_num = len(img_name_list)
    for img_name in img_name_list:

        count += 1
        if np.random.rand() > 1:
            continue

        if (count%50)==0:
            print('process finished {:.2f} percentage.'.format(
                    100*count/total_img_num))

        img_path = os.path.join(FLAGS.img_dir, img_name)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            photo_pix_npy.append(resize_image(x, 64))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pred = model.predict(x)
            embedding_vec_npy.append(pred[0])
        except Exception as e:
            print(img_name)
            pass

    embedding_vec_npy = np.asarray(embedding_vec_npy)
    photo_pix_npy = np.asarray(photo_pix_npy, dtype='uint8')

    embedding_vec_path = os.path.join(FLAGS.feat_dir, FLAGS.embedding_vec)
    photo_pix_path = os.path.join(FLAGS.feat_dir, FLAGS.photo_pix)

    np.save(embedding_vec_path+"/"+FLAGS.extract_ver, embedding_vec_npy)
    np.save(photo_pix_path+"/"+FLAGS.extract_ver, photo_pix_npy)

    print("process finished in {} seconds".format(time.time()-now))

if __name__ == '__main__':
    main(sys.argv)
