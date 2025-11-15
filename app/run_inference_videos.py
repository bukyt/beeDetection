import os
import cv2
import gdown
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, concatenate_videoclips
import requests

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------

GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/15XOsqiTSNlE4Dho9ZuhIPKQOom7R3j4N"
INPUT_DIR = "videos_input"
OUTPUT_DIR = "videos_output"
COMBINED_OUTPUT = "all_results_combined.mp4"

MODEL_PATH = r"..\runs\detect\yolov8_bee\weights\best.pt"
RESIZE_WIDTH = 1280   # resize videos before inference
RESIZE_HEIGHT = 720
# -------------------------------------------------------------


def download_first_n_from_drive(folder_url, output_dir, limit=50):
    print(f"[i] Listing Google Drive folder: {folder_url}")

    # Extract folder ID from the URL
    if "folders/" in folder_url:
        folder_id = folder_url.split("folders/")[1].split("?")[0]
    else:
        raise ValueError("Invalid Google Drive folder URL")

    # Use Google Drive API (public folders only)
    api_url = (
        f"https://www.googleapis.com/drive/v3/files"
        f"?q='{folder_id}'+in+parents&key=AIzaSyC-8exampleFAKEKEY"
    )

    # But: Google Drive blocks unauthenticated listing.
    # gdown has its own internal listing endpoint. Use that:
    list_url = f"https://drive.google.com/drive/folders/{folder_id}"

    # gdown can parse the folder webpage and extract file IDs
    file_list = gdown.download_folder(
        url=folder_url,
        output=output_dir,
        quiet=True,
        use_cookies=False,
        remaining_ok=True
    )

    # But gdown tries to download ALL files → so we must override it.
    # Instead, list files via gdown API function:

    files = gdown.download_folder(
        url=folder_url, output=None, quiet=True, remaining_ok=True, proxy=None
    )

    if not files:
        print("[!] Could not parse folder contents.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Restrict to the first N files
    files = files[:limit]

    print(f"[i] Found {len(files)} files, downloading first {limit}")

    for f in files:
        try:
            file_id = f["id"]
            filename = f["title"]
            output_path = os.path.join(output_dir, filename)

            print(f"[i] Downloading {filename}")
            gdown.download(id=file_id, output=output_path, quiet=False)
        except Exception as e:
            print(f"[!] Error downloading file {filename}: {e}")
            continue
    print("[+] Download complete (limited).")

def download_google_drive_folder(folder_url, output_dir):
    print("[i] Downloading video folder from Google Drive...")
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(folder_url, output=output_dir, quiet=False)
    print("[+] Download complete!")


def resize_video(input_file, output_file, width, height):
    cap = cv2.VideoCapture(input_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        writer.write(resized)

    cap.release()
    writer.release()


def run_yolo_inference(model, video_path, output_path):
    print(f"[i] Running detection on: {video_path}")

    name = os.path.splitext(os.path.basename(video_path))[0]

    res = model.predict(
        source=video_path,
        save=True,
        save_txt=False,
        save_conf=True,
        conf=0.25,
        imgsz=640,
        device=0,
        project="yolo_video_results",
        name=name
    )

    # YOLO writes results to: yolo_video_results/<name>/
    output_dir = os.path.join("yolo_video_results", name)

    if not os.path.exists(output_dir):
        print("[!] ERROR: YOLO output directory missing:", output_dir)
        return

    # Look for video files (YOLO can save as .mp4 or .avi)
    video_files = [f for f in os.listdir(output_dir) if f.endswith((".mp4", ".avi"))]

    if not video_files:
        print("[!] ERROR: YOLO did not save a video")
        print("    Output dir contents:", os.listdir(output_dir))
        return

    # Usually there is only one
    yolo_video = os.path.join(output_dir, video_files[0])
    
    # If YOLO saved as .avi but we need .mp4, convert it
    if yolo_video.endswith('.avi') and output_path.endswith('.mp4'):
        print(f"[i] Converting {video_files[0]} to mp4...")
        clip = VideoFileClip(yolo_video)
        clip.write_videofile(output_path, codec="libx264")
        clip.close()
        print(f"[+] Converted and saved: {output_path}")
    else:
        # Move it to your output path
        os.replace(yolo_video, output_path)
        print(f"[+] Saved annotated video: {output_path}")


def combine_videos(folder, output_file):
    print("[i] Combining videos...")
    clips = []

    for f in sorted(os.listdir(folder)):
        if f.endswith(".mp4"):
            clips.append(VideoFileClip(os.path.join(folder, f)))

    if not clips:
        print("[!] No videos to combine.")
        return

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_file, codec="libx264")
    print(f"[+] Combined video saved as {output_file}")


def main():
    # Step 1 — download all videos
    try:
        download_first_n_from_drive(GOOGLE_DRIVE_FOLDER, INPUT_DIR)
    except Exception as e:
        print(f"[!] Error downloading videos: {e}")
        pass
    # Step 2 — load YOLO model
    print("[i] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 3 — process all videos
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        input_video = os.path.join(INPUT_DIR, filename)
        resized_video = os.path.join(INPUT_DIR, "RESIZED_" + filename)
        output_video = os.path.join(OUTPUT_DIR, "DETECTED_" + filename)

        print(f"[i] Resizing: {filename}")
        resize_video(input_video, resized_video, RESIZE_WIDTH, RESIZE_HEIGHT)

        print(f"[i] Running detection: {filename}")
        run_yolo_inference(model, resized_video, output_video)

    # Step 4 — Combine all results into one video
    combine_videos(OUTPUT_DIR, COMBINED_OUTPUT)


if __name__ == "__main__":
    main()