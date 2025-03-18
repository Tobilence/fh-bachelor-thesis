import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Define source and destination directories
src_img_dir = os.path.join(os.getcwd(), '../data', 'kaggle-wood-defects', 'images')
src_label_dir = os.path.join(os.getcwd(), '../data', 'kaggle-wood-defects', 'labels-yolo')
dataset_root = os.path.join(os.getcwd(), '../data', 'wood-defects-parsed')

# Create destination directories
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(f'{dataset_root}/images/{split}', exist_ok=True)
    os.makedirs(f'{dataset_root}/labels/{split}', exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]


# Shuffle the files
random.shuffle(image_files)

# Calculate split sizes (70% train, 20% val, 10% test)
total = len(image_files)
train_size = int(0.7 * total)
val_size = int(0.1 * total)

# Split the files
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Function to copy files for a split
def copy_files(files, split):
    for f in tqdm(files):
        # Copy image
        src_img = os.path.join(src_img_dir, f)
        dst_img = os.path.join(dataset_root, 'images', split, f)
        shutil.copy2(src_img, dst_img)
        
        # Copy corresponding label file
        label_file = os.path.splitext(f)[0] + '.txt'
        src_label = os.path.join(src_label_dir, label_file)
        dst_label = os.path.join(dataset_root, 'labels', split, label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

# Copy files for each split
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print(f"Dataset split complete:")
print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")
print(f"Test: {len(test_files)} images")