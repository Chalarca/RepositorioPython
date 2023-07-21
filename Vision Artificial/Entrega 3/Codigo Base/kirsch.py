import numpy as np
import cv2

img = cv2.imread('d1.jpg')  # Leemos la imagen
height, width = img.shape[:2]  # Obtenemos sus dimensiones
img1 = np.zeros((height, width), np.uint8)
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, img1)  # Conversion a escala de grises
kirsch0 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])   # Matriz de convolucion para filtro kirsch a 0 grados
kirsch45 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])   # Matriz de convolucion para filtro kirsch a 45 grados
kirsch90 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])   # Matriz de convolucion para filtro kirsch a 90 grados
fltKrh0 = cv2.filter2D(img1, ddepth=-1, kernel=kirsch0, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fltKrh45 = cv2.filter2D(img1, ddepth=-1, kernel=kirsch45, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fltKrh90 = cv2.filter2D(img1, ddepth=-1, kernel=kirsch90, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fullEdge = fltKrh0 + fltKrh45 + fltKrh90
cv2.imshow('Imagen original', img1)
cv2.imshow('Filtro Kirsch Full', fullEdge)
cv2.imshow('Filtro Kirsch 0', fltKrh0)  # Filtro kirsch a 0 grados
cv2.imshow('Filtro Kirsch 45', fltKrh45)  # Filtro kirsch a 45 grados
cv2.imshow('Filtro Kirsch 90', fltKrh90)  # Filtro kirsch a 90 grados
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documnetacion OpenCV

