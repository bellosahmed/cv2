import os
import cv2

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY)

i = cv2.blur(thresh, (10, 10))

cv2.imshow('grey', grey)
cv2.imshow('thresh', thresh)
cv2.imshow('i', i)
cv2.waitKey(0)