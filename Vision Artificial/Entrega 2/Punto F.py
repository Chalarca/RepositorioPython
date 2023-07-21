
import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Entrega 3\Imagen\img6.png')
img2 = cv2.imread('Vision Artificial\Entrega 3\Imagen\img5.png')
img3 = cv2.imread('Vision Artificial\Entrega 3\Imagen\img8.png')
img4 = cv2.imread('Vision Artificial\Entrega 3\Imagen\img9.png')

def Areaimg (img):
    _, binari = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    PixBlan=np.sum(binari)/255
    return PixBlan

x= Areaimg(img3)
print(x)
alto,ancho=img1.shape[:2]



