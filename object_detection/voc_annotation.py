import xml.etree.ElementTree as ET
from os import getcwd

sets=[('GS', 'train'), ('GS', 'val'), ('GS', 'test')]

classes = ["upper_outer", "bottom_outer", "dress", "shoes", "hat"] # refine


def convert_annotation(appdix, image_id, list_file):
    in_file = open('VOCdevkit/VOC_%s/Annotations/%s.xml'%(appdix, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    line = ""
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text),
             int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        line += (" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    return line

wd = getcwd()

for appdix, image_set in sets:
    image_ids = open(
            'VOCdevkit/VOC_%s/ImageSets/Main/%s.txt'%(appdix, image_set)
    ).read().strip().split()

    list_file = open('%s_%s.txt'%(appdix, image_set), 'w')
    for image_id in image_ids:
        list_file.write(
                '%s/VOCdevkit/VOC_%s/JPEGImages/%s.jpg'%(wd, appdix, image_id))

        list_file.write(convert_annotation(appdix, image_id, list_file))
        list_file.write('\n')
    list_file.close()

