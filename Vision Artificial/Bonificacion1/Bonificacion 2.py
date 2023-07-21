import cv2
import time
import numpy as np

img1 = cv2.imread("Vision Artificial\Archivos\Yian_KutKu.jpg")  # Leemos la imagen
height, width = img1.shape[:2]  # Obtenemos sus dimensiones
print(img1.shape)
img2 = np.zeros((width+height, width+height, 3), np.uint8)  # Creamos una imagen nueva
img3 = np.zeros((height, width, 3), np.uint8)  # Creamos una imagen nueva

start_time = time.time()

m=np.float32([[1,0,0],[0,1,0]])
img5=cv2.warpAffine(img1,m,(width+height,height+width))

s=4
for j in range(0, width+height-3, s):
    y=j
    for i in range(0, width+height):
        y=j-i
        if y > -4 and width > i  :
            img2[y, i] = img5[y, i]
            img2[y+1, i,0] = img5[y+1, i,0]
            img2[y+2, i,1] = img5[y+2, i,1]
            img2[y+3, i,2] = img5[y+3, i,2]
        else:
            break

img3=img2[0:height,0:width]

print('Tiempo de ejecucion de ciclos:', end="")
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow('imagen ciclica', np.hstack([img1,img3]))  # Mostramos las imagenes
cv2.waitKey(0)  # Se espera a pulsar cualquier tecla para cerrar la imagen
cv2.destroyAllWindows()  # Cierre de ventanas