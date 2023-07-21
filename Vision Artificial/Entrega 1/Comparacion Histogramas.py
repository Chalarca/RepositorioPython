import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')
#img2 = cv2.imread('Vision Artificial\Archivos\Aleatron.jpg')
#img2 = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')
height, width = img1.shape[:2]
img2=img1[0:height//2,0:width//2]
#cv2.imshow("recorte",img2)
#cv2.waitKey()


colores=("b","g","r")


canales=cv2.split(img1)
canales2=cv2.split(img2)
plt.figure(1)
plt.title("Histograma A")
plt.xlabel("niveles")
plt.ylabel("# de Pixels")



for (canal, col) in zip(canales,colores):
    hist1=cv2.calcHist([canal],[0],None,[256],[0, 256])
    #cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    plt.plot(hist1, color=col)
    plt.xlim([0,256])

plt.figure(2)
plt.title("Histograma B")
plt.xlabel("niveles")
plt.ylabel("# de Pixels")

for (canal, col) in zip(canales2,colores):
    hist2=cv2.calcHist([canal],[0],None,[256],[0, 256])
    #cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    plt.plot(hist2, color=col)
    plt.xlim([0,256])


plt.show()

hist3=cv2.calcHist(img1,[0,1,2],None,[256,256,256],[0, 256,0, 256,0, 256])
hist4=cv2.calcHist(img2,[0,1,2],None,[256,256,256],[0, 256,0, 256,0, 256])
#cv2.imshow('imagen', np.hstack([img1, img2]) )
cv2.waitKey(0) 
cv2.destroyAllWindows()

Comparacion=cv2.compareHist(hist3,hist4,cv2.HISTCMP_BHATTACHARYYA)
Comparacion1=cv2.compareHist(hist3,hist4,cv2.HISTCMP_CORREL)
Comparacion2=cv2.compareHist(hist3,hist4,cv2.HISTCMP_CHISQR)
Comparacion3=cv2.compareHist(hist3,hist4,cv2.HISTCMP_INTERSECT)
print(f"Comparacion por metodo BHATTACHARYYA: ", Comparacion, " /0 indica mayor correlacion")
print(f"Comparacion por metodo Correlacion: ", Comparacion1, " /1 indica mayor correlacion")
print(f"Comparacion por metodo Chi_Cuadrado: ", Comparacion2, " /0 indica mayor correlacion")
print(f"Comparacion por metodo Interseccion: ", Comparacion3, " /entre mayor el numero mayor correlacion")


