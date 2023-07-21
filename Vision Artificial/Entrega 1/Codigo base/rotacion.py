import numpy as np
import cv2
import math

img1 = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
img2 = np.zeros((height+width, width+height, 3), np.uint8)
transMat = np.array([[math.cos(np.pi/4), -math.sin(np.pi/4), width], [math.sin(np.pi/4), math.cos(np.pi/4), width//2], [0, 0, 1]])

for i in range(0, height):
    for j in range(0, width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        rotation = np.dot(transMat, pos)  # Realizamos el producto punto entre las martices
        img2[int(rotation[0]), int(rotation[1])] = img1[i, j]  # Aplicamos las nuevas posiciones para asignar los valores de la imagen


cv2.imshow('resultado', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Fuente: Documentacion OpenCV
