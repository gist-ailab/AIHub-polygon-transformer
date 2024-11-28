import sys
sys.path.append("./")
from refer.refer import REFER
import os
import pandas as pd
from tqdm import tqdm

# Define your dataset and data root
data_root = '/media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df896/aihub_2024_datasets/'
# dataset = 'aihub_manufact'  # Change this if you're using a different dataset
dataset = 'aihub_indoor'  # Change this if you're using a different dataset
splitBy = None  # Adjust accordingly
split = 'test'  # We're focusing on test files

# Initialize the REFER object
refer = REFER(data_root, dataset, splitBy)

# Get all reference IDs for the test split
ref_ids = refer.getRefIds(split=split)

# Initialize a list to store the data
data = []

print(f"Processing {dataset} {split} split")

# Loop over each reference ID to extract information
for this_ref_id in tqdm(ref_ids):
    # Get the image ID associated with this reference
    this_img_id = refer.getImgIds(this_ref_id)
    this_img = refer.Imgs[this_img_id[0]]
    file_name = this_img['file_name'].split('.')[0]

    # Load the reference to get category_id
    ref = refer.loadRefs(this_ref_id)
    category_id = ref[0]['category_id']

    # Get all sentences associated with this reference
    ref_sent = refer.Refs[this_ref_id]
    for sent in ref_sent['sentences']:
        sentence = sent['sent']
        file_name_sent = f"{file_name}_{sentence}"
        data.append([file_name_sent, category_id])

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data, columns=['file_name', 'category_id'])

# Define the save directory and ensure it exists
save_dir = f'datasets/finetune/{dataset}_csv_output'
os.makedirs(save_dir, exist_ok=True)

# Save the DataFrame to a CSV file
csv_file_name = os.path.join(save_dir, f"{dataset}_{split}_file_sent_category.csv")
df.to_csv(csv_file_name, index=False)
print(f"Saved CSV file: {csv_file_name}")
