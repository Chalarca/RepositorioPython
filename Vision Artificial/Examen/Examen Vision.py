import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Examen\Imagen1.jpg')
img2 = cv2.imread('Vision Artificial\Examen\Imagen.jpg')

ancho,alto=img2.shape[:2]

segmento = np.zeros((alto, ancho), np.uint8)

img1G=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2G=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

_, baseb = cv2.threshold(img1G, 200, 255, cv2.THRESH_BINARY_INV)
_, muesb = cv2.threshold(img2G, 200, 255, cv2.THRESH_BINARY_INV)

#segm1=ancho/4
#segm2=alto/4

""" for i in range (0,4):
    for j in range (0,4):
        posY=segm1*j
        posX=segm2*i """

figuras,_=cv2.findContours(baseb,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
muestra, _ = cv2.findContours(muesb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for fig in muestra:
    fig = cv2.convexHull(fig)
    m = cv2.moments(fig)
    if (m["m00"]==0): m["m00"]=1
    xmuestra = int(m['m10']/m['m00'])
    ymuestra = int(m['m01']/m['m00'])

minresta = ancho*alto
for forma in figuras:
    forma = cv2.convexHull(forma)
    m = cv2.moments(forma)
    if (m["m00"]==0):
        m["m00"]=1
    x = int(m['m10']/m['m00'])
    y = int(m['m01']/m['m00'])
    comparar = baseb[x-xmuestra:x+ancho-xmuestra, y-ymuestra:y+alto-ymuestra]
    resta = np.sum((comparar-muesb)//255)
    print(resta, minresta)
    if resta < minresta:
        minresta = resta
        detection = forma
    x,y,w,h = cv2.boundingRect(detection)
    cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)   
        #posX=x
        #posY=y

    """ if segmento-muesb==0:
        yrec=posY
        xrec=posX """
    
    
    #cv2.imshow('segmentos',comparar)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
    #if muesb-baseb[]_
    





#cv2.rectangle(img1, (posX-xmuestra-10, posY-ymuestra-10), (posX+xmuestra+10, posY+ymuestra+10), (0, 255, 0), thickness=2)
cv2.imshow('segmentos',img1)
cv2.imshow('Base', baseb)
cv2.imshow('Muestra',muesb)
cv2.waitKey(0) 
cv2.destroyAllWindows() 