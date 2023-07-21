import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Archivos\Aleatron.jpg')  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
imgFsh = np.zeros((height, width), np.uint8)  # Creamos una imagen nueva
kern1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
imgFsh = cv2.filter2D(img1, ddepth=-1, kernel=kern1, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
'''
imgFsh = cv2.filter2D(imgFsh, ddepth=-1, kernel=kern1, anchor=(-1, -1), )  # Aplicamos el kernel a la imagen con la funcion filter2D
imgFsh = cv2.filter2D(imgFsh, ddepth=-1, kernel=kern1, anchor=(-1, -1), )  # Aplicamos el kernel a la imagen con la funcion filter2D
'''
cv2.imshow('Imagen original                               Imagen filtrada', np.hstack([img1, imgFsh]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documentacion OpenCV
