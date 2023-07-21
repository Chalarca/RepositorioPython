import numpy as np
import cv2


img1 = cv2.imread('Vision Artificial\Bonificacion 2\Avion1.jpg')
img2 = cv2.imread('Vision Artificial\Bonificacion 2\8.jpg')
alt, anch = img1.shape[:2]
alt1, anch1 = img2.shape[:2]
print(alt, anch)

_, bin1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)  # Umbralizacion con operador binario
_, bin2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)  # Umbralizacion con operador binario

x=0
y=0
b=0
a=0

for j in range(0, anch):
    for i in range(0, alt):
        if bin1[i,j,0]==0:
            a=i
            b=j
            break



for j in range(0, anch1):
    for i in range(0, alt1):
        if bin2[i,j,0]==0:
            x=i
            y=j
            break


print(a,b,x,y)
s=a-x
z=b-y

dist=np.sqrt(s**2+z**2)
velocidad=dist*30
print(dist, "Pixeles entre cada frame")
print(velocidad,"Pixeles por segundo")

cv2.imshow("Imagen",bin1)
cv2.imshow("Imagen2",bin2)

cv2.waitKey(0) 
cv2.destroyAllWindows()

        