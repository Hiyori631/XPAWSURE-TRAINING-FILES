import os
import shutil
import random
from collections import defaultdict

random.seed(42)

CLASSES = ["Bacterial", "Fungal", "Hotspot", "Mange"]
PUBLIC_DIR = "ProcessedDataset224"
LOCAL_DIR = "ProcessedDataset224_Local"
OUTPUT_DIR = "ProcessedDataset224_PublicLocal"

SPLIT_RATIO = 0.85  # 85% train, 15% valid from local data

def stratified_split(files, ratio):
    grouped = defaultdict(list)
    for f, cls in files:
        grouped[cls].append((f, cls))
    train, valid = [], []
    for cls, items in grouped.items():
        random.shuffle(items)
        split = int(len(items) * ratio)
        train.extend(items[:split])
        valid.extend(items[split:])
    random.shuffle(train)
    random.shuffle(valid)
    return train, valid

def collect_files(base_dir, split_name):
    result = []
    for cls in CLASSES:
        cls_dir = os.path.join(base_dir, split_name, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                result.append((fname, cls))
    return result

def copy_files(files, target_split):
    for fname, cls in files:
        src = None
        if target_split == "train":
            src = os.path.join(PUBLIC_DIR, "Train", cls, fname)
            if not os.path.exists(src):
                src = os.path.join(LOCAL_DIR, "Train", cls, fname)
        elif target_split == "valid":
            src = os.path.join(PUBLIC_DIR, "Valid", cls, fname)
            if not os.path.exists(src):
                src = os.path.join(LOCAL_DIR, "Train", cls, fname)
        elif target_split == "test":
            src = os.path.join(PUBLIC_DIR, "Test", cls, fname)
            if not os.path.exists(src):
                src = os.path.join(LOCAL_DIR, "Test", cls, fname)
        if src and os.path.exists(src):
            dst_dir = os.path.join(OUTPUT_DIR, target_split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, fname))

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

public_train = collect_files(PUBLIC_DIR, "Train")
public_valid = collect_files(PUBLIC_DIR, "Valid")
public_test = collect_files(PUBLIC_DIR, "Test")
local_train = collect_files(LOCAL_DIR, "Train")
local_test = collect_files(LOCAL_DIR, "Test")

local_train_split, local_valid_split = stratified_split(local_train, SPLIT_RATIO)

# Combine
train_files = public_train + local_train_split
valid_files = public_valid + local_valid_split
test_files = public_test + local_test

copy_files(train_files, "train")
copy_files(valid_files, "valid")
copy_files(test_files, "test")

print("=== New Merged Dataset ===")
for split in ["train", "valid", "test"]:
    print(f"\n{split.upper()}:")
    total = 0
    for cls in CLASSES:
        path = os.path.join(OUTPUT_DIR, split, cls)
        count = len(os.listdir(path)) if os.path.isdir(path) else 0
        print(f"  {cls}: {count}")
        total += count
    print(f"  Total: {total}")