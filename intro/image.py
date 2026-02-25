import os

import cv2

# read image
image_path = os.path.join('..', 'data/Bello.jpg')

img = cv2.imread(image_path)

# write image

cv2.imwrite(os.path.join('..', 'data/Bello_1.jpg'), img)

# visualize image

cv2.imshow('image', img)
cv2.waitKey(2000) #2 seconds