import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Entrega 2\Imagenes\image22.JPG', 0)  # Leemos la imagen
height, width = img1.shape  # Obtenemos sus dimensiones
imgGb = np.zeros((height, width), np.float16)  # Creamos una imagen nueva
noise = np.zeros((height, width), np.uint8)  # Creamos una imagen para almacenar el ruido
cv2.randu(noise, 255, 200)  # Creamos ruido aleatorio con la funcion randu
# donde el primer paramtro es la imagen destino, seguido el limite minimo inclusivo a generar
# y posteriormente el limite superiror exclusivo
imgNoise = img1 + noise
imgGb = cv2.GaussianBlur(imgNoise, (35,35), sigmaX=10, sigmaY=0)  # Aplicamos el kernel a la imagen con la funcion filter2D

cv2.imshow('Imagen original                                     Imagen filtrada', np.hstack([img1,noise ,imgNoise, imgGb]))  # Mostramos las imagenes

cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documentacion OpenCV
