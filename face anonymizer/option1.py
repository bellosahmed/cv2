import os
import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def process_img(img, face_detector):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    out = face_detector.detect(mp_image)

    if out.detections:
        for detection in out.detections:
            bbox = detection.bounding_box
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()


output_dir = '../intro/data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detect faces
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
face_detector = vision.FaceDetector.create_from_options(options)

if args.mode in ["image"]:
    # read image
    img = cv2.imread(args.filePath)
    img = process_img(img, face_detector)
    # save image
    cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

# Video mode
# elif args.mode in ['video']:
#     cap = cv2.VideoCapture(args.filePath)
#     ret, frame = cap.read()
#     output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
#                                    cv2.VideoWriter_fourcc(*'MP4V'),
#                                    25,
#                                    (frame.shape[1], frame.shape[0]))
#     while ret:
#         frame = process_img(frame, face_detector)
#         output_video.write(frame)
#         ret, frame = cap.read()
#
#     cap.release()
#     output_video.release()

elif args.mode in ['webcam']:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while ret:
        frame = process_img(frame, face_detector)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()