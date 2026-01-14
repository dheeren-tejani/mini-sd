from datasets import load_dataset
from PIL import Image

print("🕵️ Probing dataset structure...")
ds = load_dataset("bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images", split="train", streaming=True)

# Grab just ONE item
item = next(iter(ds))

print("\n" + "="*40)
print("DATASET ITEM STRUCTURE")
print("="*40)
print(f"KEYS AVAILABLE: {list(item.keys())}")
print("-" * 20)

# Check Image
img = item.get('image')
print(f"IMAGE COLUMN TYPE: {type(img)}")
if isinstance(img, dict) and 'bytes' in img:
    print("⚠️ WARNING: Image is raw bytes, not PIL!")

# Check Text
print("-" * 20)
for key in ['text', 'TEXT', 'caption', 'Caption']:
    if key in item:
        print(f"FOUND TEXT COLUMN: '{key}' -> '{item[key]}'")