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
# datasets = ['aihub_manufact']

if datasets[0] == 'aihub_indoor':
    # image_dir = '/media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df896/aihub_2024_datasets/indoor_test_1120/images'
    image_dir = './refer/data/aihub_refcoco_format/indoor_test_1121/images'
    # image_dir = '../indoor_80/images'
    # image_dir = '../AIHub_LAVT-RIS/refer/data/aihub_refcoco_format/indoor_test_1121/images'
elif datasets[0] == 'aihub_manufact':
    image_dir = './refer/data/aihub_refcoco_format/manufact_test_1120/images'
    # image_dir = '/media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df896/aihub_2024_datasets/manufact_test_1120/images'
else:
    image_dir = './datasets/images/mscoco/train2014'
val_test_files = pickle.load(open("data/val_test_files.p", "rb"))


# Define the directory containing your CSV files
if datasets[0] == 'aihub_indoor':
    csv_dir = 'data/aihub_csv_error_csv/indoor'  # Replace with the actual directory path
elif datasets[0] == 'aihub_manufact':
    csv_dir = 'data/aihub_csv_error_csv/manufact'  # Replace with the actual directory path
csv_files = glob.glob(f'{csv_dir}/*.csv')

# Initialize an empty dictionary to store bounding box values from all CSV files
bbox_dict = {}

# Load and combine data from all CSV files
for csv_file in csv_files:
    bbox_data = pd.read_csv(csv_file)
    
    # Determine prefix based on the file name
    prefix = "real_" if "real_" in csv_file else "syn_"
    
    # Convert filenames to the appropriate format and store in bbox_dict
    bbox_data['파일명'] = bbox_data['파일명'].apply(lambda x: f'{prefix}{x}')
    # Update bbox_dict with bbox data from this file
    bbox_dict.update(dict(zip(bbox_data['파일명'], bbox_data['bbox'])))  # Replace 'bbox_column_name' with actual column name


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
        splits = ['test']
        # splits = ['train', 'val']
        splitBy = None
    elif dataset == 'aihub_manufact':
        splits = ['test']
        # splits = ['train', 'val', 'test']
        splitBy = None

    # directory_path = '/media/sblee/e0289bbd-f18a-4b52-a657-9079fe07ec70/polyformer_manufact_vis/vis/aihub_manufact_test'  # 실제 디렉토리 경로로 변경하세요

    # # 모든 파일명을 리스트로 불러오기
    # all_filenames = os.listdir(directory_path)

    # # 기본 파일명을 기준으로 파일들을 그룹화
    # # base_id_to_files = defaultdict(set)
    # base_id_to_description_to_types = defaultdict(lambda: defaultdict(set))


    # for filename in all_filenames:
    #     if not filename.lower().endswith('.png'):
    #         continue
    #     parts = filename.split('_')
    #     if len(parts) < 4:
    #         continue  # 예상과 다른 형식의 파일명 건너뜀
    #     base_id = '_'.join(parts[:3])  # 'real_000377_000001'
        
    #     # 설명 부분과 파일 타입 추출
    #     # 예: '불을 끄는 물질을 담은 소화기를 제대로 작동시켜봐.png' 또는 '불을 끄는 물질을 담은 소화기를 제대로 작동시켜봐_gt_overlayed.png'
    #     # 설명(description)은 4번째부터, 파일 타입은 설명 뒤에 붙는 '_gt_overlayed', '_pred_overlayed' 여부에 따라 결정
    #     description_and_type = '_'.join(parts[3:]).replace('.png', '').strip()
    
    #     # 파일 타입 결정
    #     if description_and_type.endswith('gt_overlayed'):
    #         description = description_and_type[:-len('_gt_overlayed')]
    #         file_type = 'gt_overlayed'
    #     elif description_and_type.endswith('pred_overlayed'):
    #         description = description_and_type[:-len('_pred_overlayed')]
    #         file_type = 'pred_overlayed'
    #     else:
    #         description = description_and_type
    #         file_type = 'original'
        
    #     # base_id와 description에 파일 타입 추가
    #     base_id_to_description_to_types[base_id][description].add(file_type)
    #     # print(base_id_to_description_to_types)

    # # # 이제, 모든 그룹을 순회하며 세 가지 타입이 모두 존재하는지 확인
    # # base_filenames_set = set()
    # print(f"총 {len(base_id_to_description_to_types)}개의 base_id가 발견되었습니다.")

    # # 5종 이상의 설명(description)이 모두 존재하는 base_id만 세트에 추가
    # base_filenames_set = set()

    # for base_id, description_dict in base_id_to_description_to_types.items():
    #     # 5종 이상의 설명을 만족하는지 확인
    #     valid_description_count = 0
    #     for description, file_types in description_dict.items():
    #         # 각 설명마다 'original', 'gt_overlayed', 'pred_overlayed'가 모두 존재하는지 확인
    #         if {'original', 'gt_overlayed', 'pred_overlayed'}.issubset(file_types):
    #             valid_description_count += 1
    #     # 5종 이상인 경우 base_id.png 추가
    #     if valid_description_count == 5:
    #         base_filename = f"{base_id}.png"
    #         base_filenames_set.add(base_filename)

    # print(f"총 {len(base_filenames_set)}개의 base_id.png가 세트에 추가되었습니다.")


    save_dir = f'datasets/finetune/{dataset}_test_1121'
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
            # if idx % 5 == 0:
            #     pass
            # else:
            #     continue
            
            
            # if fn in base_filenames_set:
            #     continue
            # else:
            #     print(fn)

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
            if file_name_key in bbox_dict:
                print('bbox dict')
                # Update bbox value based on CSV data
                x1, y1, x2, y2 = map(int, bbox_dict[file_name_key].split(','))
                box_string = f'{x1},{y1},{x2},{y2}'
            else:
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




