"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/15
"""
import os
import pickle
import sys
import time

import gflags
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image

from feature_extraction.mobilenet import MobileNet
from feature_extraction.vgg16 import VGG16

gflags.DEFINE_string('img_dir',
                     'input_img',
                     'path of the image dir')
gflags.DEFINE_string('results_dir',
                     'results',
                     'path of the result dir')
gflags.DEFINE_string('feat_folder',
                     'extract_feature',
                     'path of the feature folder')
gflags.DEFINE_string('img_vect',
                     'img_vect',
                     'embedding img vectors')
gflags.DEFINE_string('img_name',
                     'img_name',
                     'name for images')
gflags.DEFINE_string('new_dcd_lev1',
                     'B43',
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

    img_path = os.path.join(FLAGS.img_dir, FLAGS.new_dcd_lev1+'_crop')
    print(img_path)

    start = time.time()
    for ite in os.walk(img_path):
        if len(ite[2]) == 0:
            continue
        new_dcd = ite[0].split('/')[-2]
        cate = ite[0].split('/')[-1]
        img_name_list = ite[2]

        img_vect_npy = list()
        new_img_name_list = list()
        count = 0
        total_img_num = len(img_name_list)
        for img_name in img_name_list:

            count += 1
            if np.random.rand() > 1:
                continue

            if (count%1000)==0:
                print('{}_{}: processed {:.4f} %'.format(100*count/total_img_num))

            img_path = os.path.join(
                    FLAGS.img_dir,
                    FLAGS.new_dcd_lev1,
                    new_dcd,
                    cate,
                    img_name)
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                new_img_name_list.append(img_name.split('.')[0])
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                img_vect = model.predict(x).reshape(-1)
                img_vect_npy.append(img_vect)
            except Exception as e:
                print(img_name)
                pass

        img_vect_npy = np.asarray(img_vect_npy)

        img_vect_path = os.path.join(
                FLAGS.results_dir, FLAGS.feat_folder, FLAGS.img_vect)

        img_name_path = os.path.join(
                FLAGS.results_dir, FLAGS.feat_folder, FLAGS.img_name)

        if not os.path.exists(img_vect_path):
            os.makedirs(img_vect_path)
        np.save(
                img_vect_path+"/"+FLAGS.new_dcd_lev1+"_"+new_dcd+"_"+cate,
                img_vect_npy)

        if not os.path.exists(img_name_path):
            os.makedirs(img_name_path)
        pickle.dump(
                new_img_name_list,
                open(img_name_path+"/"+FLAGS.new_dcd_lev1+"_"+new_dcd+"_"+cate, 'wb'))

    print("{}_{}: total cost {:.4f} seconds.".format(
            new_dcd, cate, time.time()-start))

if __name__ == '__main__':
    main(sys.argv)
