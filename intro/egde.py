import os
import cv2
import numpy as np

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))
img_edge = cv2.Canny(img, 100, 200)

edge_d = cv2.dilate(img_edge, np.ones((5, 5), dtype=np.uint8))

erode = cv2.erode(edge_d, np.ones((5, 5), dtype=np.uint8))

cv2.imshow('img', img)
cv2.imshow('img_edge', img_edge)
cv2.imshow('d', edge_d)
cv2.imshow('erode', erode)
cv2.waitKey(0)