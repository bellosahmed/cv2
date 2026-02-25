import os
import cv2

img = cv2.imread(os.path.join('.', 'data/Bello.jpg'))

print(img.shape)
# line
cv2.line(img, (100, 150), (300, 450), (0, 255, 0), 3)

# rectangle
cv2.rectangle(img, (100, 150), (300, 450), (0, 255, 0), 7)

# circle
cv2.circle(img, (200, 250), 45,  (0, 0, 255), -2)

# text
cv2.putText(img, 'I am', (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)