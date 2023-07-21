import numpy as np
import cv2

img = cv2.imread('Examen2.jpg')  # Leemos la imagen
height, width = img.shape[:2]  # Obtenemos sus dimensiones
img1 = np.zeros((height, width), np.uint8)
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, img1)  # Conversion a escala de grises
sobelRow = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Matriz de convolucion para filtro solbel sobre filas
sobelCol = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # Matriz de convolucion para filtro solbel sobre columnas
sobelx = cv2.Sobel(img1, cv2.CV_8U, 1, 0, ksize=3)  # Sobel en direccion x
sobely = cv2.Sobel(img1, cv2.CV_8U, 0, 1, ksize=3)  # Sobel en direccion y
sobelx = sobelx.astype('uint8')
sobely = sobely.astype('uint8')
sobelFull = cv2.Sobel(img1, cv2.CV_8U, 1, 1, ksize=1)  # Sobel en direccion y
sobelFull = sobelFull.astype('uint8')
sobelFull2 = sobelx + sobely
fltSbRow = cv2.filter2D(img1, ddepth=-1, kernel=sobelRow, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fltSbtCol = cv2.filter2D(img1, ddepth=-1, kernel=sobelCol, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D

cv2.imshow('Filtro Sobel1 (Columnas|Filas)', np.hstack([fltSbtCol, fltSbRow]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.imshow('sobelFull2', sobelFull2)
cv2.imshow('Filtro Sobel2 (Columnas|Filas)', np.hstack([sobelx, sobely]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas


# Fuente: Documnetacion OpenCV

