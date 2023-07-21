import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
_,base=cap.read()
negro = np.zeros(base.shape[:2], np.uint8)
time.sleep(1)

fondo = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
_, fondobina= cv2.threshold(fondo, 130, 255, cv2.THRESH_BINARY)
#fondob = cv2.GaussianBlur(fondo, (35,35), sigmaX=10, sigmaY=0)
hgram=cv2.calcHist(negro,[0,1,2],None,[256,256,256],[0, 256,0, 256,0, 256])

while True:

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #grayb = cv2.GaussianBlur(gray, (35,35), sigmaX=10, sigmaY=0)
    _, binar = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    x=np.sum(binar)/255
    print(x)
    #intruso=abs(grayb-fondob)
    intruso=cv2.bitwise_xor(fondobina,binar)
    hgram1=cv2.calcHist(intruso,[0,1,2],None,[256,256,256],[0, 256,0, 256,0, 256])
    Comparacion1=cv2.compareHist(hgram,hgram1,cv2.HISTCMP_INTERSECT)
    print(Comparacion1)


    cv2.imshow('video', binar)
    cv2.imshow("Fondo",fondobina)
    cv2.imshow("intruso",intruso)
    #cv2.imshow("camara",grayb)
    #cv2.imshow("fondo",fondob)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
