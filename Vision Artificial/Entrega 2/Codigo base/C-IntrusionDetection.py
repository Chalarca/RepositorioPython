# Jorge Torres Arboleda
# Sistema de detección de intrusos
# ESC para cerrar

import cv2
import numpy as np


cap = cv2.VideoCapture(0)

_ , static = cap.read() 
static = cv2.cvtColor(static, cv2.COLOR_BGR2GRAY)
static = cv2.GaussianBlur(static, (15, 15), sigmaX=0, sigmaY=0)
# cv2.imwrite('static.jpg',static)
y, x = static.shape[:2]
area = x*y
intrusionPercent = 2 #Percent of area to consider intrusion
alert = int((intrusionPercent/100)*area)

while 1:
    # Frame by frame capture
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    #Gaussian filter
    grayG = cv2.GaussianBlur(gray, (21, 21), sigmaX=0, sigmaY=0)

   
    _, thresh = cv2.threshold(static-grayG, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow('video3', thresh)
    
    if np.sum(thresh.astype(bool))>alert:
        cv2.putText(gray, text='Intruso Detectado', org=(30,y-50),
            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255,0,0),
            thickness=4, lineType=cv2.LINE_8)
        print('Intruso detectado')
        # cv2.imwrite('Alert.jpg',gray)
        # cv2.imwrite('Comparison.jpg',grayG)
        # cv2.imwrite('Detection.jpg',thresh)

    cv2.imshow('video', gray)
    # cv2.imshow('video2', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  #Press ESC to exit
        break


cap.release()
cv2.destroyAllWindows()