import os
import json
import pickle
from datetime import datetime
import glob
import shutil
from tqdm import tqdm
import csv

# 카테고리 정의 (MANUFACT_CATEGORIES 또는 INDOOR_CATEGORIES 사용)
MANUFACT_CATEGORIES = [
    # ... (기존 카테고리 정의)
]

INDOOR_CATEGORIES = [
    # ... (기존 카테고리 정의)
]

def split_annotations_for_dataset(input_base_dir_1, input_base_dir_2, output_vision_file, output_referring_file):
    # Prepare the COCO-style structure for vision annotations
    vision_annotations = {
        "info": {
            "description": "Custom COCO dataset",
            "url": "http://customdataset.org",
            "version": "1.0",
            "year": 2024,
            "contributor": "Custom Dataset Group",
            "date_created": str(datetime.now())
        },
        "images": [],
        "licenses": [
            {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}
        ],
        "annotations": [],
        # "categories": MANUFACT_CATEGORIES
        "categories": INDOOR_CATEGORIES  # 또는 MANUFACT_CATEGORIES
    }

    referring_output = []
    ref_id = 0
    image_id_counter = 0
    annotation_id_counter = 0  # Initialize annotation ID counter

    # 그룹별로 어노테이션 파일 리스트 생성
    ann_list = glob.glob(os.path.join(input_base_dir_1, "group_*", "annotation", "*.json")) + \
               glob.glob(os.path.join(input_base_dir_2, "group_*", "annotation", "*.json"))

    # 그룹별 어노테이션 파일 개수 확인을 위한 딕셔너리
    group_annotation_count = {}

    # 그룹 ID별로 어노테이션 파일을 수집
    group_annotation_files = {}
    for annotation_file in ann_list:
        # 데이터 유형 판별 ('real' 또는 'syn')
        if '/real/' in annotation_file:
            data_type = 'real'
        elif '/synthetic/' in annotation_file:
            data_type = 'syn'
        else:
            continue

        # 그룹 ID 추출
        group_dir = os.path.dirname(os.path.dirname(annotation_file))  # group_000001 directory
        group_num = os.path.basename(group_dir).split('_')[1]  # Extract '000001' from 'group_000001'
        group_id = f"{data_type}_{group_num}"

        # 그룹별로 어노테이션 파일 리스트를 저장
        if group_id not in group_annotation_files:
            group_annotation_files[group_id] = []
        group_annotation_files[group_id].append(annotation_file)

    # 그룹별로 어노테이션 파일 개수 확인 및 그룹 이름 출력
    for group_id, files in group_annotation_files.items():
        if len(files) != 5:
            print(f"그룹 {group_id}의 JSON 파일 개수: {len(files)}")
    print(f"총 그룹 개수: {len(group_annotation_files)}")
    valid_num = 0
    test_num = 0

    # 'test' 스플릿만 처리
    for group_id, files in tqdm(group_annotation_files.items()):
        split = 'test'  # 'test' 스플릿만 처리
        if split != 'test':
            continue  # 'test' 스플릿이 아닌 경우 스킵

        for annotation_file in files:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Construct image file path from annotation file path
            image_file = annotation_file.replace("/라벨링데이터/", "/원천데이터/")
            image_file = image_file.replace("/annotation/", "/rgb/")
            image_file = image_file.replace(".json", ".png")

            # Check if image file exists
            if not os.path.exists(image_file):
                print(f"Image file {image_file} does not exist. Skipping.")
                continue

            # Prepare image file name
            file_name = f"{group_id}_{os.path.basename(image_file)}"

            # Prepare vision annotation (COCO format)
            image_entry = {
                "file_name": file_name,
                "id": image_id_counter,  # Use a counter for unique image IDs
                "height": data['images']['height'],
                "width": data['images']['width'],
                "date_captured": str(datetime.now())  # Assuming current date
            }
            vision_annotations['images'].append(image_entry)

            # Process each annotation within the JSON file
            for ann in data['annotations']:
                # Generate a unique annotation ID
                unique_ann_id = annotation_id_counter
                annotation_id_counter += 1  # Increment the counter

                # Vision annotation: Include bbox, segmentation, area, etc.
                try:
                    vision_annotation = {
                        "image_id": image_id_counter,
                        "id": unique_ann_id,
                        "category_id": ann['category_id'],
                        "bbox": ann['bbox'],
                        "segmentation": ann['segmentation'],
                        "area": ann['area'],
                        "iscrowd": ann['iscrowd'],
                    }
                    if ann['bbox'] is None:
                        print(f"Annotation without bbox in file: {annotation_file}")
                        continue
                except KeyError as e:
                    print(f"Error in annotation: Missing key {e} in file {annotation_file}")
                    continue

                # Referring annotation format
                try:
                    sentences = [
                        {"raw": ann['referring_expression'], "sent_id": i, "sent": ann['referring_expression']}
                        for i in range(len([ann['referring_expression']]))
                    ]
                except KeyError:
                    print(f"Missing 'referring_expression' in annotation file: {annotation_file}")
                    continue

                vision_annotations['annotations'].append(vision_annotation)
                valid_num += 1

                referring_annotation = {
                    "ref_id": ref_id,
                    "category_id": ann['category_id'],
                    "image_id": image_id_counter,
                    "file_name": file_name,
                    "ann_id": unique_ann_id,
                    "split": split,
                    "sentences": sentences,
                    "sent_ids": [i for i in range(len(sentences))]
                }
                referring_output.append(referring_annotation)
                ref_id += 1

            image_id_counter += 1
            test_num += 1

            # Move image to the output directory
            output_image_dir = os.path.join(os.path.dirname(output_vision_file), "images")
            os.makedirs(output_image_dir, exist_ok=True)
            shutil.copy(image_file, os.path.join(output_image_dir, file_name))
        
    print("Total valid annotations: ", valid_num)
    print("Total test images: ", test_num)

    # Save the vision annotations to a JSON file
    with open(output_vision_file, 'w', encoding='utf-8') as f:
        json.dump(vision_annotations, f, ensure_ascii=False, indent=4)

    # Save the referring annotations to a pickle file
    with open(output_referring_file, 'wb') as f:
        pickle.dump(referring_output, f)

if __name__ == "__main__":
    # Set the input directories for annotations
    input_dir_1 = "refer/data/라벨링데이터/real"
    input_dir_2 = "refer/data/라벨링데이터/synthetic"

    # Set the output file paths
    output_vision_file = "refer/data/aihub_refcoco_format/manufact_test_1120/instances.json"
    output_referring_file = "refer/data/aihub_refcoco_format/manufact_test_1120/refs.p"

    # Call the function to process the dataset
    split_annotations_for_dataset(input_dir_1, input_dir_2, output_vision_file, output_referring_file)


