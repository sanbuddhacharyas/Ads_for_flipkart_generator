from __future__ import print_function
from PIL import Image
import numpy as np
import cv2, scipy
import scipy.misc
import scipy.cluster
from skimage.color import rgb2lab, deltaE_cie76
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from PIL import Image
import cv2,os,time
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def convert_rgb_to_names(rgb_tuple):
    try:
        # a dictionary of all the hex and their respective names in css3
        css3_db = CSS3_HEX_TO_NAMES
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))

        kdt_db = KDTree(rgb_values)
        distance, index = kdt_db.query(rgb_tuple)

        return names[index]
    except:
        return 'None'


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def merge_related_group(rgb_colors, hsv_colors, rgb_hsv_color_per, number_of_colors=4):
    try:
        h_threshold = 10
        s_threshold = 30
        v_threshold = 30

        # find if the colors are of same group
        for x_ind in range(number_of_colors):
            for y_ind in range(number_of_colors):
                if (x_ind < y_ind) and (y_ind < len(hsv_colors)):

                    if (abs(int(hsv_colors[x_ind][0]) - int(hsv_colors[y_ind][0])) < h_threshold) and \
                            (abs(int(hsv_colors[x_ind][1]) - int(hsv_colors[y_ind][1])) < s_threshold) and \
                            (abs(int(hsv_colors[x_ind][2]) - int(hsv_colors[y_ind][2])) < v_threshold):
                        hsv_colors[x_ind] = hsv_colors[x_ind] if (np.sum(hsv_colors[x_ind]) > np.sum(hsv_colors[y_ind])) else hsv_colors[y_ind]
                        hsv_colors.pop(y_ind)

                        rgb_colors[x_ind] = rgb_colors[x_ind] if (np.sum(rgb_colors[x_ind]) > np.sum(rgb_colors[y_ind])) else rgb_colors[y_ind]
                        rgb_colors.pop(y_ind)

                        rgb_hsv_color_per[x_ind] = rgb_hsv_color_per[x_ind] + rgb_hsv_color_per[y_ind]
                        rgb_hsv_color_per.pop(y_ind)

        return rgb_colors, hsv_colors, rgb_hsv_color_per
    except Exception as e:
        print(e)
        return [(255,255,255)],[(255,255,255)],[(255,255,255)]


def find_hsv_group(hsv_colors):
    group = []
    H_range = {'H1': (0, 10), 'H2': (10, 20), 'H3': (20, 30), 'H4': (30, 40), 'H5': (40, 50), 'H6': (50, 60),
               'H7': (60, 70), 'H8': (70, 80), 'H9': (80, 90), 'H10': (90, 100), 'H11': (100, 110), 'H12': (110, 120),
               'H13': (120, 130), 'H14': (130, 140), 'H15': (140, 150), 'H16': (150, 160), 'H17': (160, 170),
               'H18': (170, 180)}
    S_range = {'S1': (0, 15), 'S2': (15, 32), 'S3': (32, 45), 'S4': (45, 70), 'S5': (70, 91), 'S6': (91, 116),
               'S7': (116, 128), 'S8': (128, 162), 'S9': (162, 188), 'S10': (188, 206), 'S11': (206, 230),
               'S12': (230, 256)}
    V_range = {'V1': (0, 25), 'V2': (25, 52), 'V3': (52, 75), 'V4': (75, 101), 'V5': (101, 131), 'V6': (131, 167),
               'V7': (167, 206), 'V8': (206, 256)}
    for key, val in zip(H_range.keys(), H_range.values()):
        if hsv_colors[0] in range(val[0], val[1]):
            group.append(key)

    for key, val in zip(S_range.keys(), S_range.values()):
        if hsv_colors[1] in range(val[0], val[1]):
            group.append(key)

    for key, val in zip(V_range.keys(), V_range.values()):
        if hsv_colors[2] in range(val[0], val[1]):
            group.append(key)

    return group


def get_colors(image, NUM_CLUSTERS=4, show_chart=False):
    try:
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        im = im.resize((100, 100))      # optional, to reduce time
        ar = np.asarray(im)
        #ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        modified_image = ar.reshape(ar.shape[0]*ar.shape[1], 3).astype(float)
        modified_image = np.where(modified_image <= 252, modified_image, 255)

        ar = modified_image[(modified_image!=np.array([[255, 255, 255]])).all(axis=1)]

        if list(ar)==[]:
            return [(255, 255, 255)],[(255, 255, 255)], [0]

        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)         

        counts, bins = np.histogram(vecs, len(codes))   
        dic = {tuple(v):k for k,v in zip(counts,codes)}
        dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}  #sorting based on max occurance
        col = [k for k,v in dic.items()]
        bgr_equi = [np.array([dominant_color[2], dominant_color[1], dominant_color[0]],dtype='uint8').reshape(1, 1, 3) \
                    for dominant_color in col]
        hsv = [list(cv2.cvtColor(cl, cv2.COLOR_BGR2HSV).squeeze()) for cl in bgr_equi] 
        hsv = [(int(round(a)),int(round(b)),int(round(c))) for (a,b,c) in hsv]
        total_count= sum([v for k,v in dic.items()])
        per = [v*100/total_count for k,v in dic.items()]
        return col, hsv, per
    except Exception as e:
        print(e)
        return [(255, 255, 255)], [(255, 255, 255)],[0]


def skin_detector(image):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    min_YCrCb = np.array([59, 130, 74], np.uint8)
    max_YCrCb = np.array([235, 178, 123], np.uint8)
    converted_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    # cv2.imshow('YCRCB', converted_ycrcb)
    mask = cv2.inRange(converted_ycrcb, min_YCrCb, max_YCrCb)
    # cv2.imshow('YCRCB mask', skin_range)
    inverted_mask = cv2.bitwise_not(mask)
    skin_removed = cv2.bitwise_and(image, image, mask=inverted_mask)
    skin_removed[inverted_mask == 0] = 255

    # Thresh Check to revert back to original image
    total_pixels = image.shape[0] * image.shape[1]
    white_p = np.logical_and(255 == skin_removed[:, :, 0],
                             np.logical_and(255 == skin_removed[:, :, 1], 255 == skin_removed[:, :, 2]))
    num_white = np.sum(white_p)
    if num_white >= int(0.75 * total_pixels):
        skin_removed = image

    # skin_removed = cv2.cvtColor(skin_removed, cv2.COLOR_BGR2RGB)

    # cv2.imshow('skin',skin_removed)
    # cv2.waitKey(0)

    return skin_removed