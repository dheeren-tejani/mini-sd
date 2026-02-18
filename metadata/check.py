import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import re

# 1. Setup Path (Adjust if needed)
LOCAL_DATA_DIR = r"C:\Users\dheer\Coding Programs\Projects\ToyStableDiffusion\metadata\v2\LAION_AESTHETIC_512" 
ANNOTATIONS_PATH = os.path.join(LOCAL_DATA_DIR, "annotations.json")

print(f"📂 analyzing: {ANNOTATIONS_PATH}")

# 2. Load Data
with open(ANNOTATIONS_PATH, 'r') as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} image entries")

# 3. Flatten Captions & Build Frequency Map
# Some datasets have 1 caption per image, some have lists. We handle both.
all_text = []
for item in data:
    caps = item['captions']
    if isinstance(caps, list):
        all_text.extend([c.lower() for c in caps])
    else:
        all_text.append(caps.lower())

print(f"✓ Analyzed {len(all_text)} total captions")

# 4. Tokenize (Simple split)
word_counts = Counter()
for caption in all_text:
    # Remove punctuation and split
    words = re.findall(r'\b\w+\b', caption)
    word_counts.update(words)

# 5. The "Success" vs "Failure" Audit
# We compare concepts you said work well vs. ones that don't.
concepts = {
    "✅ WORKS (Nature)": ["forest", "tree", "trees", "sunset", "sky", "cloud", "clouds", "landscape", "mountain", "building", "buildings"],
    "❌ FAILS (Objects)": ["car", "cars", "vehicle", "snake", "man", "woman", "person", "dog", "cat"]
}

print(f"\n{'CONCEPT':<20} | {'COUNT':<10} | {'% OF DATASET'}")
print("-" * 50)

results = {}

for category, words in concepts.items():
    print(f"--- {category} ---")
    total_cat = 0
    for w in words:
        count = word_counts[w]
        percentage = (count / len(data)) * 100
        print(f"{w:<20} | {count:<10} | {percentage:.2f}%")
        total_cat += count
    results[category] = total_cat
    print("-" * 50)

# 6. The Verdict Logic
ratio = results["✅ WORKS (Nature)"] / (results["❌ FAILS (Objects)"] + 1)
print(f"\n📊 IMBALANCE RATIO: {ratio:.1f} to 1")

if ratio > 10:
    print("🚨 VERDICT: SEVERE DATA IMBALANCE.")
    print("   Your model sees 10 nature images for every 1 object.")
    print("   CLIP is fine, but it simply hasn't seen enough cars to learn them.")
elif ratio < 2:
    print("✅ VERDICT: BALANCED DATA.")
    print("   If counts are high but images are bad, then CLIP/U-Net training is the suspect.")
else:
    print("⚠️ VERDICT: MODERATE IMBALANCE.")
    print("   The model will struggle with objects but should eventually learn them.")