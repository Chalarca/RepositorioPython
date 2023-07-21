import numpy as np
import cv2

circbarr = cv2.imread('Vision Artificial\Entrega 3\Imagen\img6.png')
img2 = cv2.imread('Vision Artificial\Entrega 3\Imagen\img5.png')
img3 = cv2.imread('Vision Artificial\Entrega 3\Imagen\img8.png',0)
circulosm = cv2.imread('Vision Artificial\Entrega 3\Imagen\img9.png',0)
alto2,ancho2=circbarr.shape[:2]
img4=np.zeros((alto2, ancho2), np.uint8)
img1=np.zeros((alto2, ancho2), np.uint8)
img4=circulosm[12:153,0:165]
img1=circbarr[0:141,1:164]
alto,ancho=img4.shape[:2]

#print(alto,ancho,alto2,ancho2)
def Opening (img,morfologia, elemento, tamaño,iteraciones):
    height, width = img.shape[:2] 
    imgG = np.zeros((height, width), np.uint8)
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, imgG)
    _, grises = cv2.threshold(imgG, 60, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(elemento, ksize=(tamaño, tamaño))
    opening = cv2.morphologyEx(grises, morfologia, kernel, iterations=iteraciones)
    #cv2.imshow('Cam meta', grises)
    #cv2.imshow('Cam meta3', img2)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
    return opening
    


def Areaimg (inicio,meta):
    #inig=cv2.cvtColor(inicio, cv2.COLOR_BGR2GRAY)
    #metag=cv2.cvtColor(meta, cv2.COLOR_BGR2GRAY)
    #_, binari = cv2.threshold(inicio, 200, 255, cv2.THRESH_BINARY)
    #_, binari2 = cv2.threshold(meta, 200, 255, cv2.THRESH_BINARY)
    #PixBlan=np.sum(binari)/255
    PixBlan=np.count_nonzero(inicio)
    PixBlan2=np.count_nonzero(meta)
    #PixBlan2=np.sum(img3)/255
    #cv2.imshow('a 1', binari)
    #cv2.imshow('s ', binari2)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
    #diferencia=[PixBlan,PixBlan2]
    diferencia= 100-(PixBlan*100)/PixBlan2
    return diferencia

resultado1=Opening(img2,cv2.MORPH_OPEN,cv2.MORPH_ELLIPSE,3,1)
#resultado2=Opening(resultado1,cv2.MORPH_DILATE,3,1)
#kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(2, 2))
#ope = cv2.morphologyEx(resultado1, cv2.MORPH_DILATE, kernel2, iterations=2)
#x= Areaimg(ope,img3)
resultado2=Opening(img1,cv2.MORPH_OPEN,cv2.MORPH_ELLIPSE,7,1)

Caminante= Areaimg(resultado1,img3)
Circulos= Areaimg(resultado2,img4)

print(f"error caminante: ",Caminante, "Error barras y criculos: ",Circulos)

cv2.imshow("Cir y Barr procesada",resultado2)
cv2.imshow("Cir y Barr meta",img4)
cv2.imshow('Cam procesada', resultado1)
#cv2.imshow('Imagen ', ope)
cv2.imshow('Cam meta', img3)
cv2.waitKey(0) 
cv2.destroyAllWindows() 




