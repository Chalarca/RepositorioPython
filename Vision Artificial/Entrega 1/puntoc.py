import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')
alto =img.shape[0]
ancho=img.shape[1]

#rows,cols,_ = img.shape
#height, width = img.shape[:2]
rotation = cv2.getRotationMatrix2D((ancho//2, alto//2),15,0.7)
dst = cv2.warpAffine(img, rotation, (ancho+alto, ancho+alto))
cv2.imshow('Original', img)
cv2.imshow('Rotacion', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

transM = np.float32([[1, math.tan(0.20), 0], [0, 1, 0]])
dst1 = cv2.warpAffine(dst, transM, (ancho+alto, ancho+alto), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

cv2.imshow('Shearing', dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()

translation = np.float32([[1, 0,60], [0, 1, 10]])
dst2 = cv2.warpAffine(dst1, translation, (ancho+alto, ancho+alto))
cv2.imshow('traslacion', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

translation = np.float32([[1, 0, 60], [0, 1, -10]])
regreso = cv2.warpAffine(dst2, translation, (ancho+alto, ancho+alto))
cv2.imshow('traslacion inversa', regreso)
cv2.waitKey(0)
cv2.destroyAllWindows()

transM = np.float32([[1, math.tan(-0.20), 0], [0, 1, 0]])
regresoshearing = cv2.warpAffine(regreso, transM, (ancho+alto, ancho+alto), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

cv2.imshow('Shearing', regresoshearing)
cv2.waitKey(0)
cv2.destroyAllWindows()

rotation = cv2.getRotationMatrix2D((ancho//2, alto//2),0,1)
rotacioninversa = cv2.warpAffine(img, rotation, (ancho+alto, ancho+alto))
cv2.imshow('Rotacion inversa', rotacioninversa)
cv2.waitKey(0)
cv2.destroyAllWindows()