import numpy as np
import cv2

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
   
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,5)
    can = cv2.Canny(blur,50,100)
    circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1.5,10)
    
    
    cv2.imshow('No Circle', can)
        
    cam.release()
    cv2.destroyAllWindows()
