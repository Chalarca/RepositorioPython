import numpy as np
import cv2
import time


#img = cv2.imread('Vision Artificial\Entrega 3\Imagen\CodigoBarras.png')
#alto,ancho=img.shape[:2]
#barras=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#_, Barrasbin = cv2.threshold(barras, 200, 255, cv2.THRESH_BINARY_INV)

video = cv2.VideoCapture('Vision Artificial\Entrega 3\Codigos.mp4')
_,codigo=video.read()
alto,ancho=codigo.shape[:2]
recorte=np.zeros((alto,ancho),np.uint8)
while True:
    _,codigo=video.read()
    recorte=codigo[alto//2-100:alto//2+100,0:ancho//2+180]
    alto2,ancho2=recorte.shape[:2]
    #cv2.imshow('Camara', recorte)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
    barras=cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    _, Barrasbin = cv2.threshold(barras, 120, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    s=0
    grosor=[]
    Linea=[]
    for i in range(0,ancho//2+180):
        if Barrasbin[alto2//2,i]>=255:
            s=s+1
        elif Barrasbin[alto2//2,i]==0 and s<=3:
            s=0 
        elif Barrasbin[alto2//2,i]==0 and s>3:
            grosor.append(s)
            s=0
            Linea.append(i)
        else:
            0
    cv2.rectangle(codigo, (0,alto//2), (ancho//2+180,alto//2), (0,0,255))
    cv2.rectangle(codigo, (ancho//2+180,0), (ancho//2+180,alto), (0,255,0),6)
    cv2.rectangle(codigo, (5,alto//2+100), (ancho//2+180,alto//2-100), (0,255,0),6)
    cv2.putText(codigo, str(grosor)[1:-1],(0,alto-10),2,0.6,(255,0,0), 2)
    #print(grosor)

    cv2.imshow('Camara', codigo)
    cv2.imshow('Apertura', Barrasbin) # Mostramos las imagenes
    
    time.sleep(1/30)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break