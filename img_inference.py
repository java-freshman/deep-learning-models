"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/15
"""
import os
import sys
import time

import gflags
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from models.vgg16 import VGG16

gflags.DEFINE_string('img_dir',
                     'images',
                     'path of the image folder')
FLAGS = gflags.FLAGS

def load_model():
    model = VGG16(include_top=True, weights='imagenet')
    return model

def main(argv):
    FLAGS(argv)

    # load-in model
    model = load_model()

    img_name_list = os.listdir(FLAGS.img_dir)

    now = time.time()
    for img_name in img_name_list:
        img_path = os.path.join(FLAGS.img_dir, img_name)
        img = image.load_img(img_path, target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        pred = model.predict(x)
        print('{} --> {}\n {} \n\n'.format(img_name, pred.shape, pred[0]))

    print("inference finished in {} seconds".format(time.time()-now))

if __name__ == '__main__':
    main(sys.argv)
