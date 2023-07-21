import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

img = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg', 0)  # Leemos la imagen en escala de grises
_, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)  # Umbralizacion con operador binario
_, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)  # Umbralizacion con operador binario inverso
_, thresh3 = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)  # Umbralizacion con operador de truncado
_, thresh4 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)  # Umbralizacion con operador de tozero
_, thresh5 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)  # Umbralizacion con operador de tozero inverso
grayFiltered = cv2.inRange(img, 100, 155)  # Filtrado en rango

cv2.imshow('Img', img)
cv2.imshow('Binary1', grayFiltered)
cv2.waitKey(0)
cv2.destroyAllWindows()

gs = gridspec.GridSpec(3, 3)  # Creamos un grid de 3x3 con matplotlib
plt.subplot(gs[0, 0]), plt.imshow(img, 'gray')  # Graficamos la imagen
plt.title('Imagen original')  # Damos un titulo a la imagen
plt.subplot(gs[0, 1]), plt.imshow(thresh1, 'gray')  # operador binario
plt.title('Imagen umbral binaria')
plt.subplot(gs[0, 2]), plt.imshow(thresh2, 'gray')  # operador binario inverso
plt.title('Imagen binaria inversa')
plt.subplot(gs[1, 0]), plt.imshow(grayFiltered, 'gray')  # operador truncado
plt.title('Imagen umbral')
plt.subplot(gs[2, 0]), plt.imshow(thresh3, 'gray')  # operador truncado
plt.title('Imagen truncada')
plt.subplot(gs[2, 1]), plt.imshow(thresh4, 'gray')  # operador tozero
plt.title('Imagen tozero')
plt.subplot(gs[2, 2]), plt.imshow(thresh5, 'gray')  # operador tozero inverso
plt.title('Imagen tozero inversa')
plt.show()  # Se plotean los graficos

# Fuente: Documentacion OpenCV
