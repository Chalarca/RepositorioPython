import re
import numpy as np
import cv2

#203,119,134,55,43,45

img1 = cv2.imread('Vision Artificial\Entrega 2\Imagenes\img4.png',0)

def filtro(img,tamX,tamY):
    alto, ancho = img.shape[:2]
    img2 = cv2.copyMakeBorder(img,tamY,tamY,tamX,tamX,cv2.BORDER_REFLECT)
    imgfin = np.zeros(img.shape[:2], np.uint8) 
    for j in range(0,alto+tamY):
        for i in range(0,ancho+tamX):
            m = img2[j-tamY:j+tamY,i-tamX:i+tamX]
            imgfin[j-tamY,i-tamX] = np.median(m)
    return imgfin

img3=filtro(img1,1,1)
img4=filtro(img3,1,1)
cv2.imshow('Filtered1', img1)
cv2.imshow('Filtered2', img3)
cv2.imshow('Filtered3', img4)
cv2.waitKey()
cv2.destroyAllWindows()   