#!/usr/bin/env python3
# -----------------------------------------------------------
# BeeDetection Dataset Fetcher
# Downloads bee detection data from Roboflow automatically.
#
# Usage:
#   python data_fetch.py --api-key <YOUR_API_KEY>
#
# Requires:
#   pip install roboflow
# -----------------------------------------------------------

import argparse
import os
import sys
from roboflow import Roboflow


def fetch_data(api_key, project="bees-varroa", workspace="thesismodeldev-egt2b",
               version=None, format="yolov8", output_dir="../data"):
    """
    Connects to Roboflow and downloads the specified dataset directly
    into the ../data directory.
    """
    print("[i] Authenticating with Roboflow...")
    rf = Roboflow(api_key=api_key)

    # Connect to workspace
    try:
        ws = rf.workspace(workspace)
        print(f"[+] Connected to workspace: {ws.name}")
    except Exception as e:
        print(f"[-] Could not connect to workspace '{workspace}': {e}")
        sys.exit(1)

    # Get project reference
    try:
        project_ref = ws.project(project)
        print(f"[+] Found project: {project_ref.name}")
    except Exception as e:
        print(f"[-] Could not find project '{project}' in workspace '{workspace}': {e}")
        sys.exit(1)

    # Determine dataset version
    try:
        if version is None:
            versions = project_ref.versions()
            if not versions:
                print("[-] No dataset versions found.")
                sys.exit(1)
            version = versions[-1].version
            print(f"[i] Using latest version: {version}")

        abs_output = os.path.abspath(output_dir)
        os.makedirs(abs_output, exist_ok=True)

        print(f"[i] Changing working directory to: {abs_output}")
        os.chdir(abs_output)

        print(f"[i] Downloading dataset (format={format})...")
        dataset = project_ref.version(version).download(format)

        print(f"[+] Dataset successfully downloaded to: {abs_output}")

    except Exception as e:
        print(f"[-] Download failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fetch Roboflow dataset")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--workspace", default="thesismodeldev-egt2b", help="Roboflow workspace name")
    parser.add_argument("--project", default="bees-varroa", help="Project name on Roboflow")
    parser.add_argument("--version", type=int, help="Dataset version number (optional)")
    parser.add_argument("--format", default="yolov8", help="Export format (e.g., yolov8, coco, voc)")
    parser.add_argument("--output-dir", default="../data", help="Local directory to save dataset")
    args = parser.parse_args()

    fetch_data(
        api_key=args.api_key,
        project=args.project,
        workspace=args.workspace,
        version=args.version,
        format=args.format,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
