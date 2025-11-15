#!/usr/bin/env python3
# -----------------------------------------------------------
# YOLOv8 Training Script for Bee Detection
#
# Run from inside `app/`:
#   python train_yolov8.py
# -----------------------------------------------------------

import os
import sys
import argparse
from ultralytics import YOLO


def train_yolov8(dataset_path="../data", model_type="yolov8s", epochs=100, batch_size=16):
    """
    Trains a YOLOv8 model on the specified dataset.
    Automatically detects the dataset folder containing data.yaml.
    """
    abs_data_dir = os.path.abspath(dataset_path)

    if not os.path.exists(abs_data_dir):
        print(f"[-] Data directory not found: {abs_data_dir}")
        sys.exit(1)

    # Try to locate the dataset folder (one containing data.yaml)
    dataset_path = None
    for root, dirs, files in os.walk(abs_data_dir):
        if "data.yaml" in files:
            dataset_path = root
            break

    if dataset_path is None:
        print(f"[-] Could not find any dataset with 'data.yaml' inside {abs_data_dir}")
        sys.exit(1)

    print(f"[+] Found dataset: {dataset_path}")

    # Load a pre-trained YOLOv8 model
    print(f"[i] Loading model: {model_type}.pt")
    model = YOLO(f"{model_type}.pt")

    # Train the model
    print(f"[i] Starting training for {epochs} epochs on dataset: {dataset_path}")
    model.train(
        data=os.path.join(dataset_path, "data.yaml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        name="yolov8_bee_varroa_model",
        device=0  # Use GPU if available
    )

    print("[+] Training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on bee dataset")
    parser.add_argument("--dataset-path", default="../data", help="Path to dataset folder")
    parser.add_argument("--model-type", default="yolov8s", choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"], help="YOLOv8 model variant")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size during training")
    args = parser.parse_args()

    train_yolov8(
        dataset_path=args.dataset_path,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
