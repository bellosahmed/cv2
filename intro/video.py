import os

import cv2

# read video

video_path = os.path.join('..', 'data/vi.mp4')

video = cv2.VideoCapture(video_path)

cv2.VideoCapture(video_path)

# visualize video

ret = True
while ret:
    ret, frame = video.read()

    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()