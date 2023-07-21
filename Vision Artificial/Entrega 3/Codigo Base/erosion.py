import numpy as np
import cv2

img = cv2.imread('Vision Artificial\Entrega 3\Imagen\img5.png')  # Leemos la imagen
nSize = 10
height, width = img.shape[:2]  # Obtenemos sus dimensiones
imgGray = np.zeros((height, width), np.uint8)  # Creamos una imagen nueva
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, imgGray)
ret, thr = cv2.threshold(imgGray, 0, 100, cv2.THRESH_BINARY_INV)
# kernel = np.ones((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2*nSize+1, 2*nSize+1), anchor=(-1, -1))
erosion = cv2.erode(thr, kernel, iterations=1)
cv2.imshow('Img', imgGray)
cv2.waitKey(0)
cv2.imshow('Erosion', np.hstack([thr, erosion]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas