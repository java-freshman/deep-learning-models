"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/17
"""

import numpy as np
import pandas as pd

import pickle

print("start to load ")
prd_id_list = np.load("extract_feature/img_name/20180817.npy")
df = pd.read_csv("extract_feature/prd_id_dcd_lev.csv")

new_dcd_lev1_dict = dict()
new_dcd_lev2_dict = dict()
new_dcd_lev3_dict = dict()
new_dcd_lev4_dict = dict()
for prd_id in prd_id_list:

    new_dcd_lev1 = df[df['prd_id'] == prd_id].new_dcd_lev1.values[0]
    new_dcd_lev2 = df[df['prd_id'] == prd_id].new_dcd_lev2.values[0]
    new_dcd_lev3 = df[df['prd_id'] == prd_id].new_dcd_lev3.values[0]
    new_dcd_lev4 = df[df['prd_id'] == prd_id].new_dcd_lev4.values[0]

    new_dcd_lev1_dict.setdefault(new_dcd_lev1, 0)
    new_dcd_lev1_dict[new_dcd_lev1] += 1
    new_dcd_lev2_dict.setdefault(new_dcd_lev2, 0)
    new_dcd_lev2_dict[new_dcd_lev2] += 1
    new_dcd_lev3_dict.setdefault(new_dcd_lev3, 0)
    new_dcd_lev3_dict[new_dcd_lev3] += 1
    new_dcd_lev4_dict.setdefault(new_dcd_lev4, 0)
    new_dcd_lev4_dict[new_dcd_lev4] += 1

pickle.dump(new_dcd_lev1_dict, open("extract_feature/new_dcd_lev1", 'wb'))
pickle.dump(new_dcd_lev2_dict, open("extract_feature/new_dcd_lev2", 'wb'))
pickle.dump(new_dcd_lev3_dict, open("extract_feature/new_dcd_lev3", 'wb'))
pickle.dump(new_dcd_lev4_dict, open("extract_feature/new_dcd_lev4", 'wb'))