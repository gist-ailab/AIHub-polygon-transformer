import sys
sys.path.append("./")
from refer.refer import REFER
import numpy as np
from PIL import Image, UnidentifiedImageError
import random
import os
import pandas as pd
import glob
from tqdm import tqdm

import pickle
from poly_utils import is_clockwise, revert_direction, check_length, reorder_points, \
    approximate_polygons, interpolate_polygons, image_to_base64, polygons_to_string

from collections import defaultdict

max_length = 400

data_root = './refer/data'
# datasets = ['refcoco', 'refcoco+', 'refcocog']
datasets = ['aihub_indoor']

if datasets[0] == 'aihub_indoor':
    image_dir = './refer/data/aihub_refcoco_format/indoor/images'
elif datasets[0] == 'aihub_manufact':
    image_dir = './refer/data/aihub_refcoco_format/manufact/images'
else:
    image_dir = './datasets/images/mscoco/train2014'
val_test_files = pickle.load(open("data/val_test_files.p", "rb"))


combined_train_data = []

for dataset in datasets:
    if dataset == 'refcoco':
        splits = ['train', 'val', 'testA', 'testB']
        splitBy = 'unc'
    elif dataset == 'refcoco+':
        splits = ['train', 'val', 'testA', 'testB']
        splitBy = 'unc'
    elif dataset == 'refcocog':
        splits = ['train', 'val']
        splitBy = 'umd'
    elif dataset == 'aihub_indoor':
        splits = ['train', 'val', 'test']
        splitBy = None
    elif dataset == 'aihub_manufact':
        splits = ['train', 'val', 'test']
        splitBy = None


    save_dir = f'datasets/finetune/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    for split in splits:
        num_pts = []
        max_num_pts = 0
        file_name = os.path.join(save_dir, f"{dataset}_{split}.tsv")
        print("creating ", file_name)

        uniq_ids = []
        image_ids = []
        sents = []
        coeffs_strings = []
        img_strings = []

        writer = open(file_name, 'w')
        refer = REFER(data_root, dataset, splitBy)

        # if dataset == 'aihub_manufact' and split == 'val':
        #     ref_ids = refer.getRefIds(split='validation')
        # else:
        ref_ids = refer.getRefIds(split=split)
        idx = 0
        sentence_list = []
        for this_ref_id in tqdm(ref_ids):
            this_img_id = refer.getImgIds(this_ref_id)
            this_img = refer.Imgs[this_img_id[0]]
            fn = this_img['file_name']
            img_id = fn.split(".")[0].split("_")[-1]
            
            idx += 1

            # Determine the appropriate prefix for file_name_key
            prefix = fn.split(".")[0].split("_")[0] + "_"
            file_name_key = f"{prefix}{img_id}"

            # load image
            try:
                img = Image.open(os.path.join(image_dir, this_img['file_name'])).convert("RGB")
            except UnidentifiedImageError:
                print(f"Error loading image {this_img['file_name']}")
                continue

            # convert image to string
            # img_base64 = image_to_base64(img, format='png')
            img_base64 = image_to_base64(img, format='jpeg')

            # load mask
            try:
                ref = refer.loadRefs(this_ref_id)
                ref_mask = np.array(refer.getMask(ref[0])['mask'])
            except TypeError:
                print('None mask error')
                continue
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1  # 255
            annot_img = Image.fromarray(annot.astype(np.uint8), mode="P")
            annot_base64 = image_to_base64(annot_img, format='png')

            polygons = refer.getPolygon(ref[0])['polygon']

            polygons_processed = []
            for polygon in polygons:
                # make the polygon clockwise
                if not is_clockwise(polygon):
                    polygon = revert_direction(polygon)

                # reorder the polygon so that the first vertex is the one closest to image origin
                polygon = reorder_points(polygon)
                polygons_processed.append(polygon)

            polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
            polygons_interpolated = interpolate_polygons(polygons)

            polygons = approximate_polygons(polygons, 5, max_length)

            pts_string = polygons_to_string(polygons)
            pts_string_interpolated = polygons_to_string(polygons_interpolated)

            # load box
            box = refer.getRefBox(this_ref_id)  # x,y,w,h
            # Fallback to the default logic if not in combined CSV data
            if prefix == "real_":
                x, y, w, h = box
                box_string = f'{x},{y},{x + w},{y + h}'
            elif prefix == "syn_":
                x1, y1, x2, y2 = box
                box_string = f'{x1},{y1},{x2},{y2}'
            else:
                print("Image must be either real or syn")
                exit()
            # box = refer.getRefBox(this_ref_id)  # x,y,w,h
            # print(fn.split(".")[0].split("_")[0])
            # if fn.split(".")[0].split("_")[0] == "real": 
            #     x, y, w, h = box
            #     box_string = f'{x},{y},{x + w},{y + h}'
            # elif fn.split(".")[0].split("_")[0] == "syn":
            #     x1, y1, x2, y2 = box
            #     box_string = f'{x1},{y1},{x2},{y2}'
            # else:
            #     print("Image must be either real or syn")
            #     exit()

            max_num_pts = max(max_num_pts, check_length(polygons))

            num_pts.append(check_length(polygons))
            # load text
            ref_sent = refer.Refs[this_ref_id]
            for i, (sent, sent_id) in enumerate(zip(ref_sent['sentences'], ref_sent['sent_ids'])):
                uniq_id = f"{this_ref_id}_{i}"
                if sent['sent'] in sentence_list:
                    sent['sent'] = sent['sent'] + f"_{i}"
                sentence_list.append(sent['sent'])
                instance = '\t'.join(
                    [uniq_id, str(this_img_id[0]), sent['sent'], box_string, pts_string, img_base64, annot_base64,
                     pts_string_interpolated]) + '\n'
                writer.write(instance)

                if img_id not in val_test_files and split == 'train':  # filtered out val/test files
                    combined_train_data.append(instance)
        writer.close()

# random.shuffle(combined_train_data)
# file_name = os.path.join("datasets/finetune/refcoco+g_train_shuffled.tsv")
# print("creating ", file_name)
# writer = open(file_name, 'w')
# writer.writelines(combined_train_data)
# writer.close()




