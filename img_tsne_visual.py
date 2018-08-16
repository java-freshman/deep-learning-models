"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/16
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import offsetbox

from sklearn.manifold import TSNE

def plot_embedding(x_tsne, title=None):
    x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
    x_tsne = (x_tsne - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(x_tsne.shape[0]):
            dist = np.sum((x_tsne[i] - shown_images) ** 2, 1)
            if np.min(dist) <1e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [x_tsne[i]]]
            image_box = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(photos[i], cmap=plt.cm.gray_r), x_tsne[i])
            ax.add_artist(image_box)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


print("start to load the data")
x = np.load("extract_feature/embedding_vec/20180815.npy")
photos = np.load("extract_feature/photo_pix/20180815.npy")

# photos = resize_image(photos, 64)

x_tsne = TSNE(n_components=2).fit_transform(x)
print(photos.shape)
print(x_tsne.shape)

assert photos.shape[0] == x_tsne.shape[0]

plot_embedding(x_tsne)
plt.show()
