import numpy as np
import cv2

img = cv2.imread('Vision Artificial\Archivos\Aleatron.jpg')  # Leemos la imagen
height, width = img.shape[:2]  # Obtenemos sus dimensiones
img1 = np.zeros((height, width), np.uint8)
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, img1)  # Conversion a escala de grises
canny = cv2.Canny(img1, 10, 20)  # filtro canny
canny2 = cv2.Canny(img1, 10, 100)
cv2.imshow('Imagen original', img1)
cv2.imshow('Filtro Canny ', canny)  # Mostramos las imagenes
cv2.imshow('Filtro Canny2 ', canny2)
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documnetacion OpenCV

