"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/15
"""
import os
import sys
import time

import gflags
import numpy as np
import pickle

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import Model

from models.vgg16 import VGG16
from models.mobilenet import MobileNet

gflags.DEFINE_string('img_dir',
                     '/home/wutenghu/git_wutenghu/keras-yolo3/gs_img/B43_crop',
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
gflags.DEFINE_string('new_dcd',
                     'B43130301',
                     'version of the extracted feature')
FLAGS = gflags.FLAGS

def load_model(model_name='vgg16'):
    if model_name == 'vgg16':
        model = VGG16(include_top=True, weights='imagenet')
    elif model_name == 'vgg19':
        model = None
    elif model_name == 'mobilenet':
        base_model = MobileNet(include_top=True, weights='imagenet')
        model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('reshape_1').output)
    return model

def main(argv):
    FLAGS(argv)

    # load-in model
    model = load_model(model_name='mobilenet')

    img_name_list = os.listdir(os.path.join(FLAGS.img_dir, FLAGS.new_dcd))

    embedding_vec_npy = list()
    tmp_img_name_list = list()
    start = time.time()
    count = 0
    total_img_num = len(img_name_list)
    for img_name in img_name_list:

        count += 1
        if np.random.rand() > 1:
            continue

        if (count%500)==0:
            print('process finished {:.2f} percentage.'.format(
                    100*count/total_img_num))

        img_path = os.path.join(FLAGS.img_dir, FLAGS.new_dcd, img_name)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            tmp_img_name_list.append(img_name.split('.')[0])
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            embedding_vec = model.predict(x).reshape(-1)
            embedding_vec_npy.append(embedding_vec)
        except Exception as e:
            print(img_name)
            pass

    embedding_vec_npy = np.asarray(embedding_vec_npy)

    embedding_vec_path = os.path.join(FLAGS.feat_dir, FLAGS.embedding_vec)
    img_name_path = os.path.join(FLAGS.feat_dir, FLAGS.img_name)

    np.save(
            embedding_vec_path+"/"+FLAGS.new_dcd,
            embedding_vec_npy)
    pickle.dump(
            tmp_img_name_list,
            open(img_name_path+"/"+FLAGS.new_dcd, 'wb'))

    print("process finished in {} seconds".format(time.time()-start))

if __name__ == '__main__':
    main(sys.argv)
