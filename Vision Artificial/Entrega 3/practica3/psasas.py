import numpy as np
import cv2
import time


img = cv2.imread('Vision Artificial\Entrega 3\Imagen\Campo Oscuro.jpg')
alto,ancho=img.shape[:2]
barras=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, Barrasbin = cv2.threshold(barras, 200, 255, cv2.THRESH_BINARY_INV)



    

#cv2.imshow('Camara', recorte)
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
_, Barrasbin = cv2.threshold(barras, 120, 255, cv2.THRESH_BINARY_INV)
s=0
grosor=[]
Linea=[]
for i in range(0,ancho//2+180):
    if Barrasbin[alto//2,i]>=255:
        s=s+1
    elif Barrasbin[alto//2,i]==0 and s<=3:
        s=0 
    elif Barrasbin[alto//2,i]==0 and s>3:
        grosor.append(s)
        s=0
        Linea.append(i)
    else:
        0
#cv2.rectangle(codigo, (0,alto//2), (ancho//2+180,alto//2), (0,0,255))
#cv2.rectangle(codigo, (ancho//2+180,0), (ancho//2+180,alto), (0,255,0),6)
#cv2.rectangle(codigo, (5,alto//2+100), (ancho//2+180,alto//2-100), (0,255,0),6)
#cv2.putText(codigo, str(grosor)[1:-1],(0,alto-10),2,0.6,(0,0,255), 2)
#print(grosor)

cv2.imshow('Camara', img)
cv2.imshow('Apertura', Barrasbin) # Mostramos las imagenes
#cv2.imshow('Camara', recorte)
cv2.waitKey(0) 
cv2.destroyAllWindows() 