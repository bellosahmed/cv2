import os
import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = './data'
output = []
total = 0
failed = 0

print("=" * 50)
print("STARTING FACE LANDMARK EXTRACTION")
print("=" * 50)

# Check if data folder exists
if not os.path.exists(data_dir):
    print(f"ERROR: '{data_dir}' folder not found!")
    print(f"Make sure you have a 'data' folder in: {os.getcwd()}")
    exit()

emotions = sorted(os.listdir(data_dir))
print(f"Found emotions: {emotions}")
print()

for emotion_indx, emotion in enumerate(emotions):
    emotion_path = os.path.join(data_dir, emotion)

    # Skip if not a folder (e.g. .DS_Store or desktop.ini files)
    if not os.path.isdir(emotion_path):
        print(f"SKIPPING '{emotion}' — not a folder")
        continue

    image_files = os.listdir(emotion_path)
    print(f"[{emotion_indx}] Emotion: '{emotion}' — {len(image_files)} files found")

    for image_file in image_files:

        # Skip non-image files
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"  SKIP (not an image): {image_file}")
            continue

        image_path = os.path.join(emotion_path, image_file)
        total += 1

        # Try to read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"  CANNOT READ: {image_file}")
            failed += 1
            continue

        # Extract face landmarks
        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1434:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)
            print(f"  OK: {image_file}")
        else:
            print(f"  NO FACE DETECTED: {image_file} (got {len(face_landmarks)} landmarks)")
            failed += 1

    print()

# Summary
print("=" * 50)
print(f"DONE!")
print(f"  Successful: {len(output)} / {total} images")
print(f"  Failed:     {failed} / {total} images")
print("=" * 50)

# Save only if we have data
if len(output) > 0:
    np.savetxt('data.txt', np.asarray(output))
    print(f"SAVED: data.txt ({len(output)} rows)")
else:
    print("ERROR: data.txt was NOT saved — no valid faces found!")
    print()
    print("POSSIBLE FIXES:")
    print("  1. Make sure images have a clear front-facing face")
    print("  2. Make sure images are .jpg or .png format")
    print("  3. Make sure face_landmarker.task is in your project folder")
    print("  4. Try with a photo taken from your webcam or a clear portrait")