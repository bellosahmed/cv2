import os

import cv2

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))

k_size = 9
blur = cv2.blur(img, (k_size, k_size))

gau = cv2.GaussianBlur(img, (k_size, k_size), 3)

median = cv2.medianBlur(img, k_size)

cv2.imshow('img', img)
cv2.imshow('blur', blur)
cv2.imshow('gau', gau)
cv2.imshow('median', median)
cv2.waitKey(0)

# remove noise in blur