import numpy as np
import cv2
import time

start = time.time()
img = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')  # Cargamos la imagen
height, width = img.shape[:2]  # Tomamos las dimensiones de la imagen cargada
segImg = np.zeros(img.shape, np.uint8)  # Creamos una imagen vacia con las mismas dimensines que la imagen cargada

for i in range(0, height):  # Recorremos la imagen elemento por elemento
    for j in range(0, width):
        if img[i, j, 0] <= 40 and img[i, j, 1] <= 40 and img[i, j, 2] >= 200:  # Limitamos el rango de valores que deseamos obervar
            # Definimos los nuevos valores que tomara cada canal
            segImg[i, j, 0] = 255  # Canal azul
            segImg[i, j, 1] = 255  # Canal verde
            segImg[i, j, 2] = 255  # Canal rojo

print(time.time() - start)
cv2.imshow('Color', img)  # Mostramos la imagen inicial
cv2.imshow('Segmentation', segImg)  # Mostramos la imagen resultado
cv2.waitKey(0)
cv2.destroyAllWindows()

# https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
