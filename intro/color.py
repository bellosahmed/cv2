import os

import cv2

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('img', img)
cv2.imshow('rgb', rgb)
cv2.imshow('hsv', hsv)
cv2.waitKey(0)