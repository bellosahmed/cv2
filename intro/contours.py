import os
import cv2

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thres = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)> 200:
        #cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)

        x1, y1, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + w), (0, 0, 255), 2)


cv2.imshow('img', img)
cv2.imshow('thres', thres)
cv2.imshow('grey', grey)
#cv2.imshow('con', )
cv2.waitKey(0)