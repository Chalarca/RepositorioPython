import numpy as np
import cv2
import math

img1 = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
img2 = np.zeros((height*2, width*2), np.uint8)  # Creamos una imagen nueva
transMat = np.array([[1, 0, height], [0, 1, height], [0, 0, 1]])  # Creamos la matriz de transformacion

for i in range(0, height):
    for j in range(0, width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        translation = np.dot(transMat, pos)  # Realizamos el producto punto entre las martices
        img2[translation[0], translation[1]] = img1[i, j]  # Aplicamos las nuevas posiciones para asignar los valores de la imagen

cv2.imshow('resultado', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img3 = np.zeros((height*6, width*6), np.uint8)  # Creamos una imagen nueva
transMat3 = np.array([[1, math.tan(0.17), 0], [0, 1, 0], [0, 0, 1]])  # Creamos la matriz de transformacion

for i in range(0, height*6):
    for j in range(0, width*6):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        shearing = np.dot(transMat3, pos)  # Realizamos el producto punto entre las martices
        if  img2[i, j] == 0:
            pass
        else:
            img3[int(shearing[0]), int(shearing[1])] = img2[i, j]  # Aplicamos las nuevas posiciones para asignar los valores de la imagen

cv2.imshow('resultado', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

img4 = np.zeros((height*3, width*3), np.uint8)  # Creamos una imagen nueva
transMat4 = np.array([[1, math.tan(0.17), 0], [0, 1, 0], [0, 0, 1]])  # Creamos la matriz de transformacion

for i in range(0, height*2):
    for j in range(0, width*2):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        shearing2 = np.dot(transMat4, pos)  # Realizamos el producto punto entre las martices
        img4[int(shearing2[0]), int(shearing2[1])] = img3[i, j]  # Aplicamos las nuevas posiciones para asignar los valores de la imagen

cv2.imshow('resultado2', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Fuente: Documentacion
#  OpenCV
