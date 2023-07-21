import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Entrega 2\Imagenes\img3.png', 0)  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
imgFm = np.zeros((height, width), np.uint8)  # Creamos una imagen nueva
noise = np.zeros((height, width), np.uint8)
cv2.randn(noise, 0, 255)  # Creamos el ruido aleatorio con la funcion randn
# ingresando la imagen destino y los valores promedio a generar
imgNoise = img1 + noise
imgFm = cv2.medianBlur(imgNoise, 33)
cv2.imwrite('img4.png', imgNoise)
cv2.imshow('Imagen original                               Imagen filtrada', np.hstack([img1, noise, imgNoise, imgFm]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documentacion OpenCV
