import os
import shutil
import random

def split_dataset(source_dir, train_ratio, val_ratio, test_ratio):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    train_dir = os.path.join(source_dir, 'train')
    val_dir = os.path.join(source_dir, 'val')
    test_dir = os.path.join(source_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    random.shuffle(all_files)

    total_files = len(all_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]

    for f in train_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(test_dir, f))

    print(f"Total files: {total_files}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

source_directory = '/mnt/186/c/leolu030066/dataset/GFIQA-20k/image'
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

split_dataset(source_directory, train_ratio, val_ratio, test_ratio)
