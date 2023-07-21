import numpy as np
import cv2
import math
import matplotlib.pyplot as plt  

imagen1 = cv2.imread("Vision Artificial\Archivos\Yian_KutKu.jpg")
height,width = imagen1.shape[:2]
imagen2 = np.zeros((height,width,3), np.uint8)
matriztransfor = np.array([[math.cos(np.pi), -math.sin(np.pi), 0], [math.sin(np.pi), math.cos(np.pi), 0], [0, 0, 1]])

for i in range(0,height):
    for j in range(0,width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        rotacion = np.dot(matriztransfor, pos)  # Realizamos el producto punto entre las martices
        imagen2[int(rotacion[0]), int(rotacion[1])] = imagen1[i, j]
        
interpolacion = np.zeros([height, width, 3], np.uint8)
interpolacion[:,:,0] = imagen2[:,:,0]
interpolacion[:,:,1] = imagen2[:,:,1]
interpolacion[:,:,2] = imagen2[:,:,2]

for i in range(0, height):
    for j in range(0, width-1):

        if interpolacion[i,j,0]==0 and interpolacion[i,j,1]==0 and interpolacion[i,j,2]==0:
            
            interpolacion[i,j,0]= (imagen2[i,j-1,0]/2+imagen2[i,j+1,0]/2) 
        
            interpolacion[i,j,1]= (imagen2[i,j-1,1]/2+imagen2[i,j+1,1]/2) 
    
            interpolacion[i,j,2]= (imagen2[i,j-1,2]/2+imagen2[i,j+1,2]/2) 

cv2.imshow('Original', imagen1) 
cv2.imshow('imagen rotada con interpolacion',interpolacion)
cv2.waitKey(0)
cv2.destroyAllWindows()
 


shearing1= np.zeros((height*2,width*2, 3),np.uint8)
transform= np.array([[1, math.tan(0.20),0], [0, 1, 0]])

for i in range(0,height):
    for j in range(0,width):
        pos = np.array([[i], [j], [1]])
        shearing=np.dot(transform, pos)
        shearing1[int(shearing[0]), int(shearing[1])] = interpolacion[i, j]

       
cv2.imshow('shearing',shearing1)
cv2.waitKey(0)
cv2.destroyAllWindows()

traslacion1 = np.zeros((height*2, width*2, 3), np.uint8)  # Creamos una imagen nueva
transMat = np.array([[1, 0,60], [0, 1, 60]])  # Creamos la matriz de transformacion

for i in range(0, height):
    for j in range(0, width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        translation = np.dot(transMat, pos)  # Realizamos el producto punto entre las martices
        #print(translation[1])
        traslacion1[translation[0], translation[1]] = interpolacion[i, j]  # Aplicamos las nuevas posiciones para asignar los valores de la imagen

shearing2= np.zeros((height*2,width*2, 3),np.uint8)
transform= np.array([[1, math.tan(0.20),0], [0, 1, 0]])

for i in range(0,height):
    for j in range(0,width):
        pos = np.array([[i], [j], [1]])
        shearing=np.dot(transform, pos)
        shearing2[int(shearing[0]), int(shearing[1])] =  traslacion1[i, j]

       
cv2.imshow('translacion con el shearing',shearing2)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Operaciones contrarias B


traslacioninversa = np.zeros((height*2, width*2, 3), np.uint8)  # Creamos una imagen nueva
transMat = np.array([[1, 0,-60], [0, 1, -60]])  # Creamos la matriz de transformacion

for i in range(0, height):
    for j in range(0, width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        translation = np.dot(transMat, pos)  # Realizamos el producto punto entre las martices
        #print(translation[1])
        traslacioninversa[translation[0], translation[1]] = traslacion1[i, j]  # Aplicamos las nuevas posiciones para asignar los valores de la imagen


shearing3= np.zeros((height*2,width*2, 3),np.uint8)
transform= np.array([[1, math.tan(0.20),0], [0, 1, 0]])

for i in range(0,height):
    for j in range(0,width):
        pos = np.array([[i], [j], [1]])
        shearing=np.dot(transform, pos)
        shearing3[int(shearing[0]), int(shearing[1])] =  traslacioninversa[i, j]

       
cv2.imshow('traslacion inversa',shearing3)
cv2.waitKey(0)
cv2.destroyAllWindows()

traslacioninversa2 = np.zeros((height, width, 3), np.uint8)  # Creamos una imagen nueva
transMat = np.array([[1, 0,-60], [0, 1, -60]])  # Creamos la matriz de transformacion

for i in range(0, height):
    for j in range(0, width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        translation = np.dot(transMat, pos)  # Realizamos el producto punto entre las martices
        #print(translation[1])
        traslacioninversa2[translation[0], translation[1]] = traslacion1[i, j] 

cv2.imshow('shearing inverso',traslacioninversa2)
cv2.waitKey(0)
cv2.destroyAllWindows()

rotinversa = np.zeros((height,width,3), np.uint8)
matriztransfor = np.array([[math.cos(np.pi), -math.sin(np.pi), 0], [math.sin(np.pi), math.cos(np.pi), 0], [0, 0, 1]])

for i in range(0,height):
    for j in range(0,width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        rotacion = np.dot(matriztransfor, pos)  # Realizamos el producto punto entre las martices
        rotinversa[int(rotacion[0]), int(rotacion[1])] = traslacioninversa2[i, j]
        

traslacioninversa3 = np.zeros((height, width, 3), np.uint8)  # Creamos una imagen nueva
transMat = np.array([[1, 0,-60], [0, 1, -60]])  # Creamos la matriz de transformacion

for i in range(0, height):
    for j in range(0, width):
        pos = np.array([[i], [j], [1]])  # Creamos la matriz de posiciones
        translation = np.dot(transMat, pos)  # Realizamos el producto punto entre las martices
        #print(translation[1])
        traslacioninversa3[translation[0], translation[1]] = rotinversa[i, j] 



interpolacion2 = np.zeros([height, width, 3], np.uint8)
interpolacion2[:,:,0] = traslacioninversa3[:,:,0]
interpolacion2[:,:,1] = traslacioninversa3[:,:,1]
interpolacion2[:,:,2] = traslacioninversa3[:,:,2]

for i in range(0, height):
    for j in range(0, width-1):

        if interpolacion2[i,j,0]==0 and interpolacion2[i,j,1]==0 and interpolacion2[i,j,2]==0:
            
            interpolacion2[i,j,0]= (traslacioninversa3[i,j-1,0]/2+traslacioninversa3[i,j+1,0]/2) 
        
            interpolacion2[i,j,1]= (traslacioninversa3[i,j-1,1]/2+traslacioninversa3[i,j+1,1]/2) 
    
            interpolacion2[i,j,2]= (traslacioninversa3[i,j-1,2]/2+traslacioninversa3[i,j+1,2]/2)        
        
cv2.imshow('rotacion inversa',interpolacion2 )
cv2.waitKey(0)
cv2.destroyAllWindows()