import os
import argparse
import cv2


def process_img(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # blur faces
        img[y:y + h, x:x + w, :] = cv2.blur(img[y:y + h, x:x + w, :], (30, 30))

    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()

output_dir = '../intro/data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use OpenCV's built-in face detector (no external file needed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if args.mode in ["image"]:
    img = cv2.imread(args.filePath)
    img = process_img(img, face_cascade)
    cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

elif args.mode in ['webcam']:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while ret:
        frame = process_img(frame, face_cascade)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()