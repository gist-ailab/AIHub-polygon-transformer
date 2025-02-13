import json
import os
from tqdm import tqdm
import random
import pickle
import pandas as pd
import glob


img_path = 'refer/data/aihub_refcoco_format/manufact/images'

# load annotation files
# f = open("datasets/annotations/instances.json")
f = open("refer/data/aihub_refcoco_format/manufact/instances.json")
print("Loading annotation file")
data = json.load(f)
f.close()

# Initialize an empty dictionary to store bounding box values from all CSV files
bbox_dict = {}


# load the validation and test image list of refcoco, refcoco+, and refcocog
# val_test_files = pickle.load(open("data/val_test_files.p", "rb"))

# create result folder
os.makedirs("datasets/pretrain", exist_ok=True)

# generate training tsv file
print(data['annotations'][10])
print(data['annotations'][1])
print(data['annotations'][2])
print(data['annotations'][3])
print(data['annotations'][4])
print(data['annotations'][5])
print(data['annotations'][6])

# print(data['images'][0])
print(len(data['images']))
print(len(data['annotations']))

ref_file = 'refer/data/aihub_refcoco_format/manufact/refs.p'
ref_ann = pickle.load(open(ref_file, 'rb'))
print(ref_ann[10])
print(ref_ann[1])
print(ref_ann[2])
print(ref_ann[3])
print(ref_ann[4])
print(ref_ann[5])
print(ref_ann[6])

print(len(ref_ann))

# exit()


tsv_filename = "datasets/pretrain/train_aihub_manufact_80.tsv"
writer = open(tsv_filename, 'w')
print("generating ", tsv_filename)

lines = []

train_idx = 0
# ref_ann_i = next((d for d in ref_ann if d["ref_id"] == str(i)), None)
ann_id_list = [d["id"] for d in data['annotations']]
# print(ann_id_list)
# exit()
# ref_ann_i = ref_ann[i]
# for i, ann_i in enumerate(tqdm(data['annotations'])):
for i, ref_ann_i in enumerate(tqdm(ref_ann)):
    # ann_idx = ann_id_list.index(ref_ann_i["ref_id"])
    ann_i = data['annotations'][i]
    image_id = ann_i['image_id']
    bbox = ann_i['bbox']
    
    # assert ann_i['image_id'] == ref_ann_i['image_id']
    # if ann_i['image_id'] != ref_ann_i['image_id']:
    #     print(ann_i, ref_ann_i)
    if ref_ann_i["ann_id"] != ann_i["id"]:
        print(ann_i, ref_ann_i)
    
    if ref_ann_i['split'] == 'train':
        # print("train!!")
        pass
    else:
        # print(ref_ann_i['split'])
        # print("It's validation or test data")
        continue

    expressions = ref_ann_i['sentences'][0]['raw']
    # print(expressions)
    # print(expressions[0])
    
    img_dict_i = next((d for d in data['images'] if d["id"] == image_id), None)
    height, width = img_dict_i['height'], img_dict_i['width']

    # if "핸드백의 손잡이가" in expressions:
    if "\t" in expressions:
        print(expressions)
        print(img_dict_i['file_name'])
        expressions = expressions.replace('\t', '')
        print(expressions)


    try:
        fn = img_dict_i['file_name']
        img_id = fn.split(".")[0].split("_")[-1]

        # Determine the appropriate prefix for file_name_key
        prefix = fn.split(".")[0].split("_")[0] + "_"
        file_name_key = f"{prefix}{img_id}"
        # load box
        if file_name_key in bbox_dict:
            print('bbox dict')
            # Update bbox value based on CSV data
            x1, y1, x2, y2 = map(int, bbox_dict[file_name_key].split(','))
            box_string = f'{x1},{y1},{x2},{y2}'
        else:
            # prefix = img_dict_i['file_name'].split('_')[0]
            # print(prefix)
            # box = refer.getRefBox(this_ref_id)  # x,y,w,h
            # Fallback to the default logic if not in combined CSV data
            if prefix == "real_":
                x, y, w, h = bbox
                box_string = f'{x},{y},{x + w},{y + h}'
            elif prefix == "syn_":
                x1, y1, x2, y2 = bbox
                box_string = f'{x1},{y1},{x2},{y2}'
            else:
                print("Image must be either real or syn")
                exit()
    except TypeError:
        # print(bbox)
        print(ann_i)
        continue

        
    
    img_name = img_dict_i['file_name']
    filepath = os.path.join(img_path, img_name)
    
    line = '\t'.join([str(train_idx), expressions.replace('\n', ''), box_string, filepath]) + '\n'
    lines.append(line)
    train_idx += 1

# shuffle the training set
random.shuffle(lines)

# write training tsv file
writer.writelines(lines)
writer.close()

#####################################
# generate validation tsv files
# tsv_filename = f"datasets/pretrain/val_aihub_indoor_80.tsv"
tsv_filename = f"datasets/pretrain/val_aihub_manufact_80.tsv"
writer = open(tsv_filename, 'w')
print("generating ", tsv_filename)

lines = []

val_idx = 0
# for i, ann_i in enumerate(tqdm(data['annotations'])):
for i, ref_ann_i in enumerate(tqdm(ref_ann)):
    ann_i = data['annotations'][int(ref_ann_i["ref_id"])]
    image_id = ann_i['image_id']
    bbox = ann_i['bbox']
    
    # ref_ann_i = next((d for d in ref_ann if d["ref_id"] == str(i)), None)
    # ref_ann_i = ref_ann[i]
    if ref_ann_i['split'] == 'val':
        # print("val!!")
        pass
    else:
        # print("It's train or test data")
        continue

    expressions = ref_ann_i['sentences'][0]['raw']
    # print(expressions)
    # print(expressions[0])
    
    img_dict_i = next((d for d in data['images'] if d["id"] == image_id), None)
    height, width = img_dict_i['height'], img_dict_i['width']

    try:
        fn = img_dict_i['file_name']
        img_id = fn.split(".")[0].split("_")[-1]

        # Determine the appropriate prefix for file_name_key
        prefix = fn.split(".")[0].split("_")[0] + "_"
        file_name_key = f"{prefix}{img_id}"
        # load box
        if file_name_key in bbox_dict:
            print('bbox dict')
            # Update bbox value based on CSV data
            x1, y1, x2, y2 = map(int, bbox_dict[file_name_key].split(','))
            box_string = f'{x1},{y1},{x2},{y2}'
        else:
            # prefix = img_dict_i['file_name'].split('_')[0]
            # print(prefix)
            # box = refer.getRefBox(this_ref_id)  # x,y,w,h
            # Fallback to the default logic if not in combined CSV data
            if prefix == "real_":
                x, y, w, h = bbox
                box_string = f'{x},{y},{x + w},{y + h}'
            elif prefix == "syn_":
                x1, y1, x2, y2 = bbox
                box_string = f'{x1},{y1},{x2},{y2}'
            else:
                print("Image must be either real or syn")
                exit()
    except TypeError:
        # print(bbox)
        print(ann_i)
        continue
    
    img_name = img_dict_i['file_name']
    filepath = os.path.join(img_path, img_name)
    
    line = '\t'.join([str(val_idx), expressions.replace('\n', ''), box_string, filepath]) + '\n'
    lines.append(line)
    val_idx += 1

# write tsv file
writer.writelines(lines)
writer.close()

# print("train_idx", train_idx)
print('val_idx', val_idx)