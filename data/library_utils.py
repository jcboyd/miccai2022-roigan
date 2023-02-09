import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cv2 import addWeighted
from PIL import Image, ImageDraw

from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import closing, dilation, disk
from skimage.feature import blob_dog, blob_log
from skimage.color import rgb2hed, hed2rgb
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import pairwise_distances
from functools import reduce
            

def write_flush(*text_args, stream=sys.stdout):
    stream.write(', '.join(map(str, text_args)) + '\n')
    stream.flush()
    return

def get_cells(input_img, gauss_sigma, min_sigma, max_sigma, threshold):

    filtered_img = gaussian(input_img, sigma=gauss_sigma)
    inv_img = 1 - filtered_img

    blobs = blob_log(inv_img, min_sigma=min_sigma, max_sigma=max_sigma,
                     threshold=threshold, exclude_border=True)

    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    return blobs

def find_clusters(points):

    dists = pairwise_distances(points, points)

    H, W = dists.shape
    clusters = []

    def find_cluster(clusters, point):
        for cluster_idx, cluster in enumerate(clusters):
            if point in cluster:
                return cluster_idx
        return None

    for i in range(H):

        point_i = tuple(points[i])[::-1]  # PIL takes (x, y) coordinates

        for j in range(W):

            if j == i:
                continue

            if dists[i, j] <= 35:

                point_j = tuple(points[j])[::-1]  # PIL takes (x, y) coordinates
                cluster_i = find_cluster(clusters, point_i)
                cluster_j = find_cluster(clusters, point_j)

                if cluster_i is None and cluster_j is None:  # new cluster
                    clusters.append({point_i, point_j})

                elif cluster_i is None:  # i joins j
                    clusters[cluster_j].add(point_i)

                elif cluster_j is None:  # j joins i
                    clusters[cluster_i].add(point_j)

                else:  # merge
                    if cluster_i == cluster_j:
                        continue
                    merged_cluster = clusters[cluster_i].union(clusters[cluster_j])
                    cluster_i = clusters[cluster_i]
                    cluster_j = clusters[cluster_j]
                    clusters.remove(cluster_i)
                    clusters.remove(cluster_j)
                    clusters.append(merged_cluster)

    return clusters

def get_cluster_mask(clusters):

    mask = Image.new('1', (256, 256), 0)

    for cluster in clusters:
        if len(cluster) < 3:
            continue

        cluster_list = list(cluster)
        hull = ConvexHull(cluster_list)
        ordered_points = [tuple(cluster_list[idx]) for idx in hull.vertices]
        ImageDraw.Draw(mask).polygon(ordered_points, outline=1, fill=1)

    return np.array(mask)

def decompose_ihc(input_img):

    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(input_img)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    return ihc_h, ihc_e, ihc_d

def get_crops_image(img, cells, crop_size=48):

    crops = []
    bboxes = []

    for cell in cells:
        y, x = map(int, cell)
        top, bottom = y - crop_size // 2, y + crop_size // 2
        left, right = x - crop_size // 2, x + crop_size // 2
        crop = img[top:bottom, left:right]

        if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
            crops.append(crop)
            bboxes.append([left, top, right, bottom])

    return crops, bboxes
