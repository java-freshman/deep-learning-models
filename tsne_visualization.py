import os

import numpy as np
import scipy.spatial.distance as distance
import pandas as pd
import random
import matplotlib.pyplot as plt

import time

from keras.preprocessing import image



prd_img_vgg16 = np.load('extract_feature/embedding_vec/20180817.npy')
prd_id_list = np.load('extract_feature/img_name/20180817.npy')
df = pd.read_csv('extract_feature/prd_id_map_dcd_lev.csv')
prd_id_img_vgg16 = dict()

for idx in range(len(prd_id_list)):
    prd_id_img_vgg16[prd_id_list[idx]] = prd_img_vgg16[idx]

prd_id_dcd_lev4 = dict()
for dcd_lev4 in set(df.new_dcd_lev4.values):
    prd_id_dcd_lev4[dcd_lev4] = set(df[df.new_dcd_lev4 == dcd_lev4].prd_id.values).intersection(set(prd_id_list))

for dcd_lev4 in prd_id_dcd_lev4:
    print(dcd_lev4, len(prd_id_dcd_lev4[dcd_lev4]))


import pickle

count = 0
for dcd_lev4 in prd_id_dcd_lev4:
    prd_id_similar = dict()
    prd_id_set = prd_id_dcd_lev4[dcd_lev4]
    if len(prd_id_set) <= 1:
        continue
    print("start the {} category".format(dcd_lev4))
    print("current category has {} items".format(len(prd_id_set)))
    for prd_id_a in prd_id_set:
        a = prd_id_img_vgg16[prd_id_a]
        tmp_dict = dict()
        for prd_id_b in prd_id_set:
            if prd_id_b == prd_id_a:
                continue
            b = prd_id_img_vgg16[prd_id_b]
            tmp_dict[prd_id_b] = 1 - distance.cosine(a, b)

        prd_id_similar[prd_id_a] = sorted(tmp_dict.items(), key=lambda t:t[1])[-10:]
        count += 1
    
    print("finish {} items".format(count))

    pickle.dump(prd_id_similar, open(dcd_lev4+'_prd_id_similar_dict', 'wb'))

