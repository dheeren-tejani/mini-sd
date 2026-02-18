import os
from PIL import Image
import json
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
import io
import torch
import re
from collections import Counter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR = "LAION_AESTHETIC_512"
TARGET_SIZE = 512
MAX_IMAGES = None  # Download ALL (~167k)
# ==============================================================================

IMAGES_DIR_OUT = os.path.join(OUTPUT_DIR, "images")
ANNOTATIONS_FILE_OUT = os.path.join(OUTPUT_DIR, "annotations.json")
PROCESSED_FLAG_OUT = os.path.join(OUTPUT_DIR, ".processed_flag")

os.makedirs(IMAGES_DIR_OUT, exist_ok=True)

print("="*60)
print("STARTING LOCAL LAION PROCESSING (Fixed for Byte Data)")
print(f"Output Folder: {OUTPUT_DIR}")
print("="*60)

# --- 1. Load Dataset ---
print(f"[1/4] Connecting to Hugging Face...")
ds = load_dataset("bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images", split="train", streaming=True)

annotations = []
processed_count = 0
failed_count = 0

# --- 2. Process Images ---
print(f"[2/4] Processing images...")
pbar = tqdm(ds, desc="Processing", total=167000)

for item in pbar:
    if MAX_IMAGES is not None and processed_count >= MAX_IMAGES:
        break

    try:
        img_data = item.get('image')
        
        if isinstance(img_data, dict) and 'bytes' in img_data:
            # Unwrap the bytes and convert to PIL
            image = Image.open(io.BytesIO(img_data['bytes']))
        elif isinstance(img_data, Image.Image):
            image = img_data
        else:
            raise ValueError(f"Unknown image format: {type(img_data)}")

        # Check Text
        caption_text = item.get('text', item.get('caption', ""))
        if not caption_text: continue

        # Resize
        image = image.convert('RGB')
        image = image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
        
        # Save
        new_filename = f"image_{processed_count:06d}.jpg"
        save_path = os.path.join(IMAGES_DIR_OUT, new_filename)
        image.save(save_path, quality=95)
        
        annotations.append({
            "image_id": processed_count,
            "image_filename": new_filename,
            "captions": [caption_text],
            "width": TARGET_SIZE,
            "height": TARGET_SIZE
        })
        
        processed_count += 1
        
    except Exception as e:
        failed_count += 1
        continue

print(f"\n✓ Successfully processed {len(annotations)} images.")

# --- 3. Save Annotations ---
print(f"[3/4] Saving JSON...")
with open(ANNOTATIONS_FILE_OUT, 'w') as f:
    json.dump(annotations, f, indent=2)

# --- 4. Build Tokenizer ---
print(f"[4/4] Building Tokenizer...")

# Tokenizer Class
class SimpleTokenizer:
    def __init__(self, vocab_size=49408):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, captions):
        word_freq = Counter()
        for caption in tqdm(captions, desc="Tokenizing"):
            words = re.findall(r'\b\w+\b', caption.lower())
            word_freq.update(words)
        
        most_common = word_freq.most_common(self.vocab_size - len(self.word2idx))
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

# Build
all_captions = [item["captions"][0] for item in annotations]
tokenizer = SimpleTokenizer(vocab_size=49408)
tokenizer.build_vocab(all_captions)

torch.save({
    "word2idx": tokenizer.word2idx,
    "idx2word": tokenizer.idx2word
}, os.path.join(OUTPUT_DIR, "tokenizer.pt"))

print(f"✓ Tokenizer saved.")
print("VICTORY! Now zip the folder and upload to Drive.")