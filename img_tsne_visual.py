"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/16
"""

import numpy as np
import pandas as pd

import time

import tensorflow as tf
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

from utils.visual_utils import plot_clustering, plot_embedding


def resize_image(array, image_size):
    resize_tensor = tf.image.resize_images(
            array,
            (image_size, image_size),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session().as_default():
        result = resize_tensor.eval()
    return result


print("start to load ")
x = np.load("extract_feature/embedding_vec/20180817.npy")
prd_id_list = np.load("extract_feature/img_name/20180817.npy")
df = pd.read_csv("extract_feature/prd_id_map_dcd_lev.csv")

now = time.time()
x_tsne = TSNE(n_components=3, n_jobs=8).fit_transform(x)
print(time.time()-now)
print(prd_id_list.shape)
print(x_tsne.shape)

assert prd_id_list.shape[0] == x_tsne.shape[0]

labels = list()
for prd_id in prd_id_list:
    labels.append(df[df['prd_id']==prd_id].new_dcd_lev2.values[0])
    # print(df[df['prd_id']==prd_id].new_dcd_lev2.values[0])

plot_clustering(x=x_tsne, labels=labels, num_labels=8.)
