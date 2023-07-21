import cv2
import numpy as np
img = cv2.imread('Vision Artificial\Archivos\RGB.png')

#LAB
min=np.array([0,0,0])
max=np.array([122,130,86])

frameLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
mask = cv2.inRange(frameLAB, min, max)
cv2.imshow('conv',img)
cv2.imshow('maskLABvis', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()