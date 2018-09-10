"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/30
"""
import os
from yolo import YOLO
from PIL import Image

from timeit import default_timer as timer

img_root_folder = "gs_img"
dcd_folder = "B43"

new_dcd_classes_dict = {
    'B43130517': {'sweater', 'blouse', 'shirt', 't_shirt', 'polo_shirt'},  # t shirt
    'B43130701': {'dress', 'skirt', 'pants'},  # pants
    'B43130509': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt', 'polo_shirt'},    # blouse
    'B43130301': {'dress', 'skirt', 'pants'},  # dress
    'B43130501': {'sweater', 'blouse', 'shirt', 't_shirt', 'polo_shirt'},  # shirt
    'B43130519': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt',
                  'polo_shirt'},  # cardigan
    # 'B43050103': {},  # bra/panty set
    # 'B43050107': {},  # panties
    # 'B43050501': {},  # socks
    # 'B43070903': {},  # swimsuit
    # 'B43130503': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt',
    #               'polo_shirt'},  # knit/sweater
    # 'B43130511': {'coat', 'sweater', 'blouse', 'shirt', 't_shirt',
    #               'polo_shirt'},  # jacket
    # 'B43130703': {'dress', 'skirt', 'pants'}  # skirt
}

def detect_img(yolo):

    cwd = os.getcwd()
    new_dcd_path = os.path.join(cwd, img_root_folder, dcd_folder)
    crop_new_dcd_path = new_dcd_path + "_crop"

    if not os.path.exists(crop_new_dcd_path):
        os.mkdir(crop_new_dcd_path)

    new_dcd_list = os.listdir(new_dcd_path)

    for new_dcd in new_dcd_list:
        print(new_dcd)
        start = timer()

        img_path = os.path.join(new_dcd_path, new_dcd)
        img_list = os.listdir(img_path)
        class_set = new_dcd_classes_dict[new_dcd]

        crop_img_path = os.path.join(crop_new_dcd_path, new_dcd)
        if not os.path.exists(crop_img_path):
            os.mkdir(crop_img_path)

        count = 0
        for img in img_list:
            if count % 1000 == 0:
                print("{} progressed {:.4f} %".format(new_dcd, count*100/len(img_list)))
            try:
                image = Image.open(img_path+'/'+img)
                r_image = yolo.detect_image(image, img, crop_img_path, class_set)
            except Exception as e:
                print(img_path)
            count += 1

        end = timer()
        print("Total cost {} seconds.".format(end - start))
        yolo.close_session()

def main():
    detect_img(yolo=YOLO())

if __name__ == '__main__':
    main()
