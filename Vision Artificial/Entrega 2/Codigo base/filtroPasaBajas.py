import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Archivos\Aleatron.jpg', 0)  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
imgFpb = np.zeros((height, width), np.uint8)  # Creamos una imagen nueva
kern1 = np.ones((3, 3),np.float32)/9  # Creamos la matriz del kernel
imgFpb = cv2.filter2D(img1, ddepth=-1, kernel=kern1, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
cv2.imshow('Imagen original                               Imagen filtrada', np.hstack([img1, imgFpb]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documentacion OpenCV
