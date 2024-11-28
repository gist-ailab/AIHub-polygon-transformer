import sys
import os
import pandas as pd
import glob
from tqdm import tqdm
import pickle

# Ensure the current directory is in the system path
sys.path.append("./")

# Import the REFER class from your module
from refer.refer import REFER

# Define utility functions if needed (assuming they are in poly_utils)
# from poly_utils import is_clockwise, revert_direction, check_length, reorder_points, \
#     approximate_polygons, interpolate_polygons, image_to_base64, polygons_to_string

# Configuration
data_root = './refer/data'
datasets = ['aihub_indoor']  # Modify as needed: ['refcoco', 'refcoco+', 'refcocog', 'aihub_indoor', 'aihub_manufact']

# Define image directories based on dataset
if datasets[0] == 'aihub_indoor':
    # image_dir = './refer/data/aihub_refcoco_format/indoor_80/images'
    # image_dir = '/media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df896/aihub_2024_datasets/indoor_test_1120/images'
    image_dir = '../AIHub_LAVT-RIS/refer/data/aihub_refcoco_format/indoor_test_1121/images'
elif datasets[0] == 'aihub_manufact':
    # image_dir = './refer/data/aihub_refcoco_format/manufact_80/images'
    image_dir = '/media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df896/aihub_2024_datasets/manufact_test_1120/images'
else:
    image_dir = './datasets/images/mscoco/train2014'

# Load val/test files if necessary (optional for mapping)
val_test_files = pickle.load(open("data/val_test_files.p", "rb"))

# Define the directory containing your CSV files (optional for mapping)
if datasets[0] == 'aihub_indoor':
    csv_dir = 'data/aihub_csv_error_csv/indoor'  # Replace with the actual directory path
elif datasets[0] == 'aihub_manufact':
    csv_dir = 'data/aihub_csv_error_csv/manufact'  # Replace with the actual directory path
csv_files = glob.glob(f'{csv_dir}/*.csv')

# Initialize an empty dictionary to store bounding box values from all CSV files (optional for mapping)
bbox_dict = {}

# Load and combine data from all CSV files (optional for mapping)
for csv_file in csv_files:
    bbox_data = pd.read_csv(csv_file)
    
    # Determine prefix based on the file name
    prefix = "real_" if "real_" in csv_file else "syn_"
    
    # Convert filenames to the appropriate format and store in bbox_dict
    bbox_data['파일명'] = bbox_data['파일명'].apply(lambda x: f'{prefix}{x}')
    # Update bbox_dict with bbox data from this file
    bbox_dict.update(dict(zip(bbox_data['파일명'], bbox_data['bbox'])))  # Replace 'bbox' with actual column name if different

# Initialize a list to store (id, file_name) mappings
id_file_mapping = []

# Iterate through each dataset
for dataset in datasets:
    # Define splits and splitBy based on dataset
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
        splits = ['test']  # Modify as needed
        splitBy = None
    elif dataset == 'aihub_manufact':
        splits = ['val', 'test']  # Modify as needed
        splitBy = None
    else:
        print(f"Unknown dataset: {dataset}")
        continue

    # Initialize REFER object
    refer = REFER(data_root, dataset, splitBy)

    # Iterate through each split
    for split in splits:
        print(f"Processing Dataset: {dataset}, Split: {split}")
        # Get all reference IDs for the current split
        ref_ids = refer.getRefIds(split=split)

        # Iterate through each reference ID with a progress bar
        for this_ref_id in tqdm(ref_ids, desc=f"{dataset} {split}"):
            # Get image IDs associated with the reference ID
            this_img_ids = refer.getImgIds(this_ref_id)
            if not this_img_ids:
                print(f"No image IDs found for ref_id: {this_ref_id}")
                continue
            this_img_id = this_img_ids[0]  # Assuming one image per ref_id

            # Get image details
            this_img = refer.Imgs[this_img_id]
            fn = this_img['file_name']

            # Append the (id, file_name) mapping
            id_file_mapping.append({'id': this_ref_id, 'file_name': fn})

# Convert the mapping list to a pandas DataFrame
mapping_df = pd.DataFrame(id_file_mapping)

# Remove duplicates if any
mapping_df.drop_duplicates(subset=['id', 'file_name'], inplace=True)

# Define the output CSV file path
output_csv_path = 'id_file_mapping.csv'

# Save the DataFrame to a CSV file
mapping_df.to_csv(output_csv_path, index=False)

print(f"ID to File Name mapping has been saved to {output_csv_path}")
