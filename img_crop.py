"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/30
"""
import os
import sys
import gflags

from PIL import Image
from timeit import default_timer as timer

from object_detection.yolo import YOLO

gflags.DEFINE_string('img_dir',
                     'input_img',
                     'path of the image dir')
gflags.DEFINE_string('new_dcd_lev1',
                     'B43',
                     'folder for store new_dcd_lev1')
FLAGS = gflags.FLAGS

new_dcd_classes_dict = {
    'B43130517': {'sweater', 'blouse', 'shirt', 't_shirt', 'polo_shirt'},  # t shirt
    'B43130701': {'dress', 'skirt', 'pants'},  # pants
    'B43130509': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt', 'polo_shirt'},  # blouse
    'B43130301': {'dress', 'skirt', 'pants'},  # dress
    'B43130501': {'sweater', 'blouse', 'shirt', 't_shirt', 'polo_shirt'},  # shirt
    'B43130519': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt',
                  'polo_shirt'},  # cardigan
    'B43050103': {},  # bra/panty set
    'B43050107': {},  # panties
    'B43050501': {},  # socks
    'B43070903': {},  # swimsuit
    'B43130503': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt',
                  'polo_shirt'},  # knit/sweater
    'B43130511': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt',
                  'polo_shirt'},  # jacket
    'B43130703': {'dress', 'skirt', 'pants'}  # skirt
}


def detect_img(yolo):
    new_dcd_path = os.path.join(FLAGS.img_dir, FLAGS.new_dcd_lev1)
    crop_new_dcd_path = new_dcd_path + "_crop"

    if not os.path.exists(crop_new_dcd_path):
        os.mkdir(crop_new_dcd_path)

    for ite in os.walk(new_dcd_path):
        if len(ite[2]) == 0:
            continue

        start = timer()
        new_dcd = ite[0].split('/')[-2]
        cate = ite[0].split('/')[-1]
        img_list = ite[2]

        if new_dcd not in new_dcd_classes_dict:
            continue

        class_set = new_dcd_classes_dict[new_dcd]
        if len(class_set) == 0:
            continue

        img_path = os.path.join(new_dcd_path, new_dcd, cate)
        crop_img_path = os.path.join(crop_new_dcd_path, new_dcd, cate)

        if not os.path.exists(crop_img_path):
            os.makedirs(crop_img_path)

        count = 0
        for img in img_list:
            if count % 1000 == 0:
                print("{}_{}: progressed {:.4f} %".format(
                    new_dcd, cate, count * 100 / len(img_list)))
            try:
                image = Image.open(img_path + '/' + img)
                yolo.detect_image(image, img, crop_img_path, class_set)
            except Exception as e:
                print(img_path + '/' + img)
            count += 1
        end = timer()
        print("{}_{}: total cost {:.4f} seconds".format(new_dcd, cate, end - start))

    yolo.close_session()


def main(argv):
    FLAGS(argv)
    detect_img(yolo=YOLO())


if __name__ == '__main__':
    main(sys.argv)
