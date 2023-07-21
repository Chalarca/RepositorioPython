import numpy as np
import cv2

img = cv2.imread('/mnt/Documents/images/12.jpg')  # Leemos la imagen
height, width = img.shape[:2]  # Obtenemos sus dimensiones
img1 = np.zeros((height, width), np.uint8)
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, img1)  # Conversion a escala de grises
rbtsRow = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])   # Matriz de convolucion para filtro roberts sobre filas
rbtsCol = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])    # Matriz de convolucion para filtro roberts sobre columnas
fltRbtRow = cv2.filter2D(img1, ddepth=-1, kernel=rbtsRow, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fltRbtCol = cv2.filter2D(img1, ddepth=-1, kernel=rbtsCol, anchor=(-1, -1))  # Aplicamos el kernel a la imagen con la funcion filter2D
fullEdge = fltRbtCol + fltRbtRow
cv2.imshow('img', img)
cv2.imshow('imgFulledge', fullEdge)
cv2.imshow('Filtro Roberts (Columnas|Filas)', np.hstack([fltRbtCol, fltRbtRow]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Fuente: Documnetacion OpenCV

