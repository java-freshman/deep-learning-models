"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/17
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import offsetbox

def plot_embedding(x, photos, title=None):
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(x.shape[0]):
            dist = np.sum((x[i] - shown_images) ** 2, 1)
            if np.min(dist) <1e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [x[i]]]
            image_box = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(photos[i], cmap=plt.cm.gray_r), x[i])
            ax.add_artist(image_box)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


# Visualize the clustering
def plot_clustering(x, labels, num_labels, title=None):
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    plt.figure(figsize=(6, 4))
    for i in range(x.shape[0]):
        plt.scatter(x[i, 0], x[i, 1], color=plt.cm.nipy_spectral(labels[i]/num_labels))
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()
    plt.show()