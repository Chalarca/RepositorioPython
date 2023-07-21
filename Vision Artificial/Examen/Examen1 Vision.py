import numpy as np
import cv2


img1 = cv2.imread('Vision Artificial\Examen\Imagen1.jpg')
img2 = cv2.imread('Vision Artificial\Examen\Imagen.jpg')

ancho,alto=img2.shape[:2]

img1G=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2G=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

_, baseb = cv2.threshold(img1G, 200, 255, cv2.THRESH_BINARY_INV)
_, muesb = cv2.threshold(img2G, 200, 255, cv2.THRESH_BINARY_INV)



formas, _ = cv2.findContours(baseb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
muestra, _ = cv2.findContours(muesb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for fig in muestra:
    m = cv2.moments(fig)
    if (m["m00"]==0):
        m["m00"]=1
    xmuestra = int(m['m10']/m['m00'])
    ymuestra = int(m['m01']/m['m00'])        


minresta = ancho*alto
for forma in formas:
    m = cv2.moments(forma)
    if (m["m00"]==0):
        m["m00"]=1
    x = int(m['m10']/m['m00'])
    y = int(m['m01']/m['m00'])
    #cv2.circle(img1, (x,y), 3, (0, 0, 255), -1)
    comparar = baseb[x-xmuestra:x+ancho-xmuestra, y-ymuestra:y+alto-ymuestra]
    resta = np.sum((comparar-muesb)//255)
    print(resta, minresta)
    if resta < minresta:
        minresta = resta
        detection = forma
    #cv2.imshow('segmentos',comparar)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 


x,y,w,h = cv2.boundingRect(detection)
cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)        
# cv2.drawContours(img1,[detection],0, (0,0,255), 2)     

        
cv2.imshow('Original', img1)
cv2.imshow('Figura', img2)
cv2.waitKey(0) 
cv2.destroyAllWindows()