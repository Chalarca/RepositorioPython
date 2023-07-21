import numpy as np
import cv2

img = cv2.imread('d1.jpg')  # Leemos la imagen
height, width = img.shape[:2]  # Obtenemos sus dimensiones
img1 = np.zeros((height, width), np.uint8)
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, img1)  # Conversion a escala de grises
prewittRow = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Matriz de convolucion para filtro prewitt sobre filas
prewittCol = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # Matriz de convolucion para filtro prewitt sobre columnas
fltPwtRow = cv2.filter2D(img1, ddepth=-1, kernel=prewittRow, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fltPwtCol = cv2.filter2D(img1, ddepth=-1, kernel=prewittCol, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fullEdge =  fltPwtCol + fltPwtRow
cv2.imshow('Filtro Prewitt (Original)', img1)  # Mostramos las imagenes
cv2.imshow('Filtro Prewitt (Columnas)', fltPwtCol)  # Mostramos las imagenes
cv2.imshow('Filtro Prewitt (Filas)', fltPwtRow)  # Mostramos las imagenes
cv2.imshow('Filtro Prewitt (Full)', fullEdge)  # Mostramos las imagenes

cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documnetacion OpenCV

