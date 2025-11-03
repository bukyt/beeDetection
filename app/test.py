#!/usr/bin/env python3
# -----------------------------------------------------------
# YOLOv8 BeeDetection Testing Script
#
# Runs inference on the test dataset and evaluates accuracy.
# Automatically finds the latest trained model in ../runs/detect/
# -----------------------------------------------------------

import os
import sys
import yaml
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def find_dataset(data_dir="../data"):
    """Find the dataset folder containing data.yaml."""
    abs_data_dir = os.path.abspath(data_dir)
    for root, _, files in os.walk(abs_data_dir):
        if "data.yaml" in files:
            return root
    return None


def load_yaml_classes(data_yaml_path):
    """Load class names from data.yaml."""
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def find_latest_model(run_dir="../runs/detect"):
    """Find the latest trained YOLOv8 model in ../runs/detect/*/weights/best.pt"""
    abs_run_dir = os.path.abspath(run_dir)
    if not os.path.exists(abs_run_dir):
        print(f"[-] Run directory not found: {abs_run_dir}")
        return None

    # Find all subfolders
    subdirs = [d for d in Path(abs_run_dir).iterdir() if d.is_dir()]
    if not subdirs:
        return None

    # Sort by modification time (latest first)
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    for d in subdirs:
        model_path = d / "weights" / "best.pt"
        if model_path.exists():
            return str(model_path)

    return None


def test_model(data_dir="../data", conf=0.25):
    """
    Tests a trained YOLOv8 model on the test dataset.
    Saves results and prints accuracy statistics.
    """
    print("[i] Searching for dataset...")
    dataset_path = find_dataset(data_dir)
    if not dataset_path:
        print("[-] Could not find dataset (missing data.yaml)")
        sys.exit(1)

    data_yaml = os.path.join(dataset_path, "data.yaml")
    names = load_yaml_classes(data_yaml)
    print(f"[+] Classes: {names}")

    test_img_dir = os.path.join(dataset_path, "test", "images")
    if not os.path.exists(test_img_dir):
        print(f"[-] Test image folder not found: {test_img_dir}")
        sys.exit(1)

    model_path = find_latest_model()
    if model_path is None:
        print("[-] Could not find any trained YOLOv8 model in ../runs/detect/")
        sys.exit(1)

    print(f"[i] Loading model from {model_path}")
    model = YOLO(model_path)

    result_dir = os.path.abspath(os.path.join(data_dir, "results"))
    os.makedirs(result_dir, exist_ok=True)

    print(f"[i] Running inference on test set...")
    results = model.predict(source=test_img_dir, conf=conf, save=True, project=result_dir, name="preds")

    # Simple accuracy estimate: compare predicted vs actual class per image
    correct, wrong, total = 0, 0, 0
    label_dir = os.path.join(dataset_path, "test", "labels")

    for res in results:
        img_path = Path(res.path)
        label_path = os.path.join(label_dir, img_path.stem + ".txt")
        if not os.path.exists(label_path):
            continue

        total += 1
        gt_labels = np.loadtxt(label_path, usecols=[0], ndmin=1).astype(int)
        pred_labels = np.array([int(b.cls) for b in res.boxes])

        # Check if any predicted class matches ground truth
        if any(cls in gt_labels for cls in pred_labels):
            correct += 1
        else:
            wrong += 1

    print("-----------------------------------------------------------")
    print(f"[+] Total images tested: {total}")
    print(f"[+] Correct detections:  {correct}")
    print(f"[+] Wrong detections:    {wrong}")
    print(f"[+] Accuracy:            {correct / total * 100:.2f}%")
    print("-----------------------------------------------------------")
    print(f"[i] Saved result images in: {os.path.join(result_dir, 'preds')}")


if __name__ == "__main__":
    test_model()
