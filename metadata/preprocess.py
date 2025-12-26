import os
from PIL import Image
import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

# ==============================================================================
# CONFIGURATION (Verify these match your folder names!)
# ==============================================================================

# Input Paths
RAW_IMAGES_DIR = "train2017"   # Folder containing the 118k .jpg files
ANNOTATIONS_PATH = "annotations/captions_train2017.json" # The captions file

# Output Path (Will be created)
OUTPUT_DIR = "COCO_PROCESSED_128"
# ==============================================================================

# Setup output directories
IMAGES_DIR_OUT = os.path.join(OUTPUT_DIR, "images")
ANNOTATIONS_FILE_OUT = os.path.join(OUTPUT_DIR, "annotations.json")
PROCESSED_FLAG_OUT = os.path.join(OUTPUT_DIR, ".processed_flag")

os.makedirs(IMAGES_DIR_OUT, exist_ok=True)

print("="*60)
print("STARTING LOCAL MS-COCO PROCESSING")
print(f"Input Images:   {RAW_IMAGES_DIR}")
print(f"Input Captions: {ANNOTATIONS_PATH}")
print(f"Output Folder:  {OUTPUT_DIR}")
print("="*60)

# --- 1. Parse Captions ---
print(f"\n[1/3] Parsing captions from {ANNOTATIONS_PATH}...")
try:
    with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {ANNOTATIONS_PATH}")
    print("   Make sure you unzipped the annotations file correctly!")
    exit()

# Map image_id -> list of captions
image_id_to_captions = defaultdict(list)
for ann in tqdm(coco_data['annotations'], desc="Reading captions"):
    image_id_to_captions[ann['image_id']].append(ann['caption'])

# Map image_id -> filename (e.g., 123 -> '000000000123.jpg')
image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

print(f"✓ Found {len(image_id_to_filename)} images in annotations file.")

# --- 2. Process Images ---
print(f"\n[2/3] Processing & Resizing images to 128x128...")
annotations = []
failed_count = 0
processed_count = 0

# Get list of actual files on disk to compare
files_on_disk = set(os.listdir(RAW_IMAGES_DIR))

for image_id, filename in tqdm(image_id_to_filename.items(), desc="Processing"):
    # Check if file exists on disk
    if filename not in files_on_disk:
        # This happens sometimes if the zip wasn't extracted fully
        continue

    try:
        # 1. Open
        input_path = os.path.join(RAW_IMAGES_DIR, filename)
        image = Image.open(input_path).convert('RGB')
        
        # 2. Resize (The "Squish")
        # Lanczos is best for downsampling (making smaller)
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
        
        # 3. Save
        # We rename them to simple indices: image_000000.jpg
        new_filename = f"image_{processed_count:06d}.jpg"
        save_path = os.path.join(IMAGES_DIR_OUT, new_filename)
        image.save(save_path, quality=95)
        
        # 4. Add to new annotations list
        annotations.append({
            "image_id": processed_count,
            "original_id": image_id,
            "image_filename": new_filename,
            "captions": image_id_to_captions.get(image_id, [""]),
            "width": 128,
            "height": 128
        })
        
        processed_count += 1

    except Exception as e:
        failed_count += 1
        if failed_count <= 10: # Only show first few errors
            print(f"⚠ Error processing {filename}: {e}")
        continue

print(f"\n✓ Successfully processed {len(annotations)} images.")
if failed_count > 0:
    print(f"⚠ Failed to process {failed_count} images (corrupt files?).")

# --- 3. Save Final Data ---
print(f"\n[3/3] Saving final {ANNOTATIONS_FILE_OUT}...")
with open(ANNOTATIONS_FILE_OUT, 'w') as f:
    json.dump(annotations, f, indent=2)

# Create the flag file so Colab knows it's done
stats = f"""Processed on: {datetime.now().isoformat()}
Original Source: MS-COCO 2017 Train
Total Images: {len(annotations)}
Image Size: 128x128
"""
with open(PROCESSED_FLAG_OUT, 'w') as f:
    f.write(stats)

print("\n" + "="*60)
print("VICTORY! DATASET READY.")
print(f"Location: {os.path.abspath(OUTPUT_DIR)}")
print("Steps to finish:")
print("1. Zip this 'COCO_PROCESSED_128' folder.")
print("2. Name it 'coco_128.zip'.")
print("3. Upload to Google Drive: /ToyStableDiffusion/")
print("="*60)