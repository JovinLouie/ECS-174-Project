import os
import shutil
from sklearn.model_selection import train_test_split

# Input folders
image_dir = "roadsegmentation-boston-losangeles/images"
mask_dir = "roadsegmentation-boston-losangeles/groundtruth"

# Output folders
output_dir = "output"
splits = ["train", "val", "test"]

for split in splits:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# Get all image filenames
images = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
 
# Split: train/val/test
trainval, test = train_test_split(images, test_size=0.15, random_state=123)
train, val = train_test_split(trainval, test_size=0.15/0.85, random_state=123)
# (This yields 70% train, 15% val, 15% test)

def copy_pairs(file_list, split):
    for fname in file_list:
        img_src = os.path.join(image_dir, fname)

        # Try to find the mask with matching basename but any extension
        base = os.path.splitext(fname)[0]

        # Check all possible mask extensions
        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
            mask_candidate = os.path.join(mask_dir, base + ext)
            if os.path.exists(mask_candidate):
                mask_src = mask_candidate
                break
        else:
            print(f"WARNING: No mask found for {fname}")
            continue

        img_dst = os.path.join(output_dir, split, "images", fname)
        mask_dst = os.path.join(output_dir, split, "masks", os.path.basename(mask_src))

        shutil.copy(img_src, img_dst)
        shutil.copy(mask_src, mask_dst)

# Copy into final directories
copy_pairs(train, "train")
copy_pairs(val, "val")
copy_pairs(test, "test")

print("Paired train/val/test split complete.")
