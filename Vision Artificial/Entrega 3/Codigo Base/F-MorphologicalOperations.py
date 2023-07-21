import numpy as np
import cv2

img = cv2.imread('Vision Artificial\Entrega 3\Imagen\img6.png')  # Leemos la imagen
objective = cv2.imread('Vision Artificial\Entrega 3\Imagen\img9.png', 0)  # Leemos la imagen
# cv2.imshow('Image', img)
cv2.imshow('Objective', objective)
nSize = 4
height, width = img.shape[:2]  # Obtenemos sus dimensiones
imgGray = np.zeros((height, width), np.uint8)  # Creamos una imagen nueva
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, imgGray)
ret, thr = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY)
# thr=imgGray
# kernel = np.zeros((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(2*nSize+1, 2*nSize+1))
opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow('Imagen 1', np.hstack([thr, opening]))  # Mostramos las imagenes
print(np.count_nonzero(objective))
print(np.count_nonzero(opening))

img = cv2.imread('Vision Artificial\Entrega 3\Imagen\img5.png')  # Leemos la imagen
objective = cv2.imread('Vision Artificial\Entrega 3\Imagen\img8.png', 0) 
# cv2.imshow('Objective', objective)
nSize = 1
height, width = img.shape[:2]  # Obtenemos sus dimensiones
imgGray = np.zeros((height, width), np.uint8)  # Creamos una imagen nueva
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, imgGray)
ret, thr = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY)
# thr=imgGray
# kernel = np.zeros((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(2*nSize+1, 2*nSize+1))
opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow('Imagen 2', np.hstack([thr, opening]))  # Mostramos las imagenes

print(np.count_nonzero(objective))
print(np.count_nonzero(opening))
print(opening.shape)
print(objective.shape)
# cv2.imshow('asdf',asdf)

cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas