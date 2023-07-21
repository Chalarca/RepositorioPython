import cv2  
import time
import numpy as np

img1 = cv2.imread("D:\Proyectos Visual Studio\Python-Jupyther\Fund_Ap_Maq_Vision\Lena.png")  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
print(img1.shape)
img2 = np.zeros((height, width, 3), np.uint8)  # Creamos una imagen nueva
img3 = np.zeros((height, width, 3), np.uint8)  # Creamos una imagen nueva

# Imagen ciclica
start_time = time.time()

# Seccion a color
y=0
s=4

for j in range(0, height, s):
    y=j
    for i in range(0, height):
        y=j-i
        if y > 0 and width > i  :
            img2[y, i] = img1[y, i]
        else:
            break
for j in range(height-width+3, height, s):
    y=j
    for z in range(0, height+width):
        f=width-z-1
        y=j+z
        if  y<height and f<width:
            img2[y, f] = img1[y, f]
        else:
            break
#componente azul

for j in range(1, height, s):
    y=j
    for i in range(0, height):
        y=j-i
        if y > 0 and width > i  :
            img2[y, i,0] = img1[y, i,0]
        else:
            break
for j in range(height-width, height, s):
    y=j
    for z in range(0, height+width):
        f=width-z-1
        y=j+z
        if  y<height and f<width:
            img2[y, f,0] = img1[y, f,0]
        else:
            break
    
#componente roja
for j in range(2, height, s):
    y=j
    for i in range(0, height):
        y=j-i
        if y > 0 and width > i  :
            img2[y, i,1] = img1[y, i,1]
        else:
            break
for j in range(height-width+1, height, s):
    y=j
    for z in range(0, height+width):
        f=width-z-1
        y=j+z
        if  y<height and f<width:
            img2[y, f,1] = img1[y, f,1]
        else:
            break

#Componente Verde

for j in range(3, height, s):
    y=j
    for i in range(0, height):
        y=j-i
        if y > 0 and width > i  :
            img2[y, i,2] = img1[y, i,2]
        else:
            break
for j in range(height-width+2, height, s):
    y=j
    for z in range(0, height+width):
        f=width-z-1
        y=j+z
        if  y<height and f<=width:
            img2[y, f,2] = img1[y, f,2]
        else:
            break
print('Tiempo de ejecucion de ciclos:', end="")
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow('imagen ciclica', np.hstack([img1, img2]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas