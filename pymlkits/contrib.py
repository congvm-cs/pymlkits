# contrib.py

import numpy as np
import os
import math
import cv2
from glob import glob
from tqdm import tqdm
import urllib
from .multiprocessing import pool_worker
from PIL import Image, ImageDraw


def upsample(fp, lbl, THRESHOLD, cutoff=False):
    """ Upsampling dataset
    """
    if not isinstance(fp, np.ndarray):
        fp = np.array(fp)

    if not isinstance(lbl, np.ndarray):
        lbl = np.array(lbl)

    class_dict = {}
    for class_id in tqdm(np.unique(lbl)):
        path_filter = fp[lbl == class_id]
        class_dict[class_id] = list(path_filter)

    new_class_dict = class_dict.copy()
    for class_id, files in class_dict.items():
        if len(files) != 0:
            if (len(files) > THRESHOLD):
                if (cutoff == True):
                    new_class_dict[class_id] = np.random.choice(
                        files, THRESHOLD)
                else:
                    new_class_dict[class_id] = files
            else:
                n_missing = THRESHOLD - len(files)
                duplicated_samples = np.random.choice(files, n_missing)
                new_class_dict[class_id].extend(duplicated_samples)
    ret_fp = []
    ret_lbl = []
    for class_id, files in new_class_dict.items():
        ret_fp.extend(files)
        ret_lbl.extend([class_id]*len(files))
    return ret_fp, ret_lbl


def get_onehot_label(y):
    return np.argmax(y, axis=1)


def load_image_url(url, proxies='10.40.34.14:81'):
    img = None
    try:
        req = urllib.request.Request(url)
        if proxies is not None:
            req.set_proxy(proxies, 'http')
        response = urllib.request.urlopen(req)
        img = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img, 1)  # cv2.IMREAD_COLOR
    except Exception as e:
        errorMessage = '{}: {}'.format(url, e)
        print(errorMessage)
    return img


def get_top_k(pred_y, top_k):
    arg_sort_y_pred = np.argsort(pred_y, )
    return arg_sort_y_pred[:, -top_k:][:, ::-1], np.asarray([pr[ar_pr][-top_k:][::-1] for pr, ar_pr in zip(pred_y, arg_sort_y_pred)])


def onehot_normalize(class_id, num_classes):
    class_id = int(class_id)
    onehot_lbl = np.zeros(shape=[num_classes, ])
    onehot_lbl[class_id] = 1
    return onehot_lbl


def multilabel_normalize(class_id, num_classes):
    multi_lbl = np.zeros(shape=[num_classes, ])
    for clc in class_id:
        multi_lbl[int(clc)] = 1
    return multi_lbl


def load_txt_multiprocessing(txt_file, parsing_func, up_sampling_threshold=None,
                             cutoff=False, num_worker=None):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    ret = pool_worker(parsing_func,
                      inputs=lines,
                      num_worker=num_worker)
    labels = []
    fnames = []
    for name, lbl in ret:
        if name is not None:
            fnames.append(name)
            labels.append(lbl)

    if up_sampling_threshold != None:
        fnames, labels = upsample(
            fnames, labels, up_sampling_threshold, cutoff)
    ret_labels = np.array(labels)
    ret_fnames = np.array(fnames)
    return ret_fnames, ret_labels


# def median_filter(img_arr):
#     img_arr = cv2.medianBlur(img_arr[..., 0], 9)
#     img_arr = img_arr[..., np.newaxis]
#     return np.concatenate([img_arr]*3, axis=2)


# def load_raw_data(img_path):
#     img = Image.open(img_path)
#     img_np = np.array(img)
#     rgb = img_np[:640, :, :3]
#     depth_original = img_np[640:, :, :]
#     depth = depth_original.reshape((-1, 4)).astype(np.uint16)
#     depth_rawmemory = np.bitwise_or(np.left_shift(
#         depth[:, 3], 8), depth[:, 0]).reshape(depth_original.shape[:2])
#     depth_map = depth_rawmemory.view(dtype=np.float16)
#     depth_map[np.isnan(depth_map)] = 0
#     return rgb, depth_map


# def crop_with_face_det(fdet_model, rgb, depth_map):
#     face_crop = None
#     depth_crop = None
#     box = [0, 0, 0, 0]
#     has_face = 0
#     boxes, labels, probs = fdet_model.detect(rgb)
#     if len(boxes) > 0:
#         box = square_bbox(rgb, boxes[0])
#         box = expand_bbox(rgb, box, ratio=1.05)
#         face_crop = crop_img(rgb, box)
#         depth_crop = crop_img(depth_map, box)
#         has_face = 1
#     return face_crop, depth_crop, box, has_face
