import os

import cv2

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))

resize_img = cv2.resize(img, (400, 400))

print(img.shape)
print(resize_img.shape)

cv2.imshow('img', img)
cv2.imshow('resize_img', resize_img)
cv2.waitKey(0)