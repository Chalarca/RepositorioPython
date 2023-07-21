import numpy as np
import cv2

imgA = cv2.imread('Vision Artificial\Entrega 2\Imagenes\image22.JPG', 0)  # Leemos la imagen
height, width = imgA.shape  # Obtenemos sus dimensiones
imgB = np.zeros((height, width), np.float16)  # Creamos una imagen nueva

imgB = cv2.GaussianBlur(imgA, (35,35), sigmaX=10, sigmaY=0)  # Aplicamos el kernel a la imagen con la funcion filter2D

imgC = (0.3*(imgA)).astype(np.uint8) + (0.7*(imgB)).astype(np.uint8) -35 #el -35 es para ajustar un poco la intensidad 

cv2.imshow("Imagen A                                                                                                         Imagen B                                                                                                                   Imagen C", 
np.hstack([imgA, imgB, imgC]))

cv2.waitKey(0) 
cv2.destroyAllWindows()

