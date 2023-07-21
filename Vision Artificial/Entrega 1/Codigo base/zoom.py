import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
img2 = np.zeros((height*2, width*2, 3), np.uint8)  # Creamos una imagen nueva
transMat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])  # Creamos la matriz de transformacion

for i in range(0, height):
    for j in range(0, width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        scaling = np.dot(transMat, pos)  # Realizamos el producto punto entre las martices
        img2[int(scaling[0]), int(scaling[1])] = img1[i, j]  # Aplicamos las nuevas posiciones para asignar los valores de la imagen

cv2.imshow('resultado', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()