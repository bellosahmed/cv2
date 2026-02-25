import os
import cv2

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#ret, thresh = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY)

adapt = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 21, 30)

ret, simple_thresh = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
#cv2.imshow('thresh', thresh)
cv2.imshow('adapt', adapt)
cv2.imshow('simple_thresh', simple_thresh)
cv2.waitKey(0)