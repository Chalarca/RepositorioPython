import cv2
import numpy as np
import time


imagen1 = cv2.imread("C:/Users/User/Desktop/vison artificial/practica3/avion1.png",0)
height1,width1 = imagen1.shape[:2]
imagen2 = cv2.imread("C:/Users/User/Desktop/vison artificial/practica3/avion2.png",0)
height2,width2 = imagen2.shape[:2]

start_time = time.time()
ret, thresh2 = cv2.threshold (imagen1, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold (imagen2, 127, 255, cv2.THRESH_BINARY_INV)
count1 = 0
count2 = 0
for i in range  (0,height1):
    for j in range (0,width1):
        if imagen1[i, j] == 255 :
            pos1 = [i, j]
            count1 =  count1 + 1
        

for i in range  (0,height2):
    for j in range (0,width2):
        if imagen2[i, j] == 255 :
            pos2 = [i, j]
            count2 =  count2 + 1     



cv2.imshow("1",thresh2)
cv2.imshow("2",thresh3)

cv2.waitKey(0)
cv2.destroyAllWindows()
