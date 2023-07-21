import cv2  
import time
import numpy as np

img1 = cv2.imread('D:\Proyectos Visual Studio\Python-Jupyther\Vision Artificial\Archivos\Aleatron.jpg')  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
print(img1.shape)
img2 = np.zeros((height, width, 3), np.uint8)  # Creamos una imagen nueva
img3 = np.zeros((height, width, 3), np.uint8)  # Creamos una imagen nueva

# Imagen ciclica
start_time = time.time()

# Seccion a color
for i in range(0, height//2):
    for j in range(0, width//2):
        img2[i, j] = img1[i, j]
# Componente azul
for i in range(0, height//2):
    for j in range(width//2, width):
        img2[i, j, 0] = img1[i, j, 0]
# Componente verde
for i in range(height//2, height):
    for j in range(0, width//2):
        img2[i, j, 1] = img1[i, j, 1]
# Componente roja
for i in range(height//2, height):
    for j in range(width//2, width):
        img2[i, j, 2] = img1[i, j, 2]

print('Tiempo de ejecucion de ciclos:', end="")
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow('imagen ciclica', np.hstack([img1, img2]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas

# Imagen vectorizada
start_time = time.time()
img3[0:height//2, 0:width//2] = (img1[0:height//2, 0:width//2])  # Seccion a color
img3[0:height//2, width//2:width, 0] = (img1[0:height//2, width//2:width, 0])  # Componente azul
img3[height//2:height, 0:width//2, 1] = (img1[height//2:height, 0:width//2, 1])  # Componente verde
img3[height//2:height, width//2:width, 2] = (img1[height//2:height, width//2:width, 2])  # Componente roja
print('Tiempo de ejecucion de vectores:', end="")
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow('imagen_vectorizada', np.hstack([img1, img3]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas


