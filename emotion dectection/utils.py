# curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions


def get_face_landmarks(image, draw=False, model_path="face_landmarker.task"):

    # Convert BGR to RGB
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Wrap in MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_input_rgb)

    # Configure FaceLandmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    image_landmarks = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)

        if results.face_landmarks:
            ls_single_face = results.face_landmarks[0]

            if draw:
                image_rows, image_cols, _ = image.shape
                # Draw connections
                for connection in [(0,1),(1,2)]:  # replace with your connection set if needed
                    pass
                # Draw landmark points
                for lm in ls_single_face:
                    x = int(lm.x * image_cols)
                    y = int(lm.y * image_rows)
                    cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=2)

            xs_ = [lm.x for lm in ls_single_face]
            ys_ = [lm.y for lm in ls_single_face]
            zs_ = [lm.z for lm in ls_single_face]

            min_x, min_y, min_z = min(xs_), min(ys_), min(zs_)

            for x, y, z in zip(xs_, ys_, zs_):
                image_landmarks.append(x - min_x)
                image_landmarks.append(y - min_y)
                image_landmarks.append(z - min_z)

    return image_landmarks