#splits dataset to train test and val
import os
import random
import shutil
from pathlib import Path

def split_dataset(root, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    images_dir = Path(root) / "train" / "images"
    labels_dir = Path(root) / "train" / "labels"

    # new dirs
    split_dirs = {
        "train": (Path(root) / "train" / "images", Path(root) / "train" / "labels"),
        "valid": (Path(root) / "valid" / "images", Path(root) / "valid" / "labels"),
        "test": (Path(root) / "test" / "images", Path(root) / "test" / "labels")
    }

    # create dirs
    for img_dir, lbl_dir in split_dirs.values():
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    def move_files(file_list, split_name):
        img_dest, lbl_dest = split_dirs[split_name]
        for img_path in file_list:
            label_path = labels_dir / (img_path.stem + ".txt")

            shutil.move(str(img_path), img_dest / img_path.name)
            if label_path.exists():
                shutil.move(str(label_path), lbl_dest / label_path.name)

    move_files(train_imgs, "train")
    move_files(val_imgs, "valid")
    move_files(test_imgs, "test")

    print("Split complete:")
    print(f"Train: {len(train_imgs)}")
    print(f"Val:   {len(val_imgs)}")
    print(f"Test:  {len(test_imgs)}")

if __name__ == "__main__":
    split_dataset("../data/bee-2")
