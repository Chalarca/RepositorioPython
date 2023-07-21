import numpy as np
import cv2

#img1 = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg') #Mi imagen de perfil favorita porque puedo >:(
img1 = cv2.imread('Vision Artificial\Archivos\RGB.png',17) #Imagen RGB para comprobar funcionamiento

img=img1.copy()

drawing = True
mode = False
ix, iy = -1, -1
count=0
r=20
def draw(event, x, y, flags, param): # Se declara la funcion
    global ix, iy, drawing, mode, count  # Defino unas variables globales

    if event == cv2.EVENT_LBUTTONDOWN:  # Se pregunta si se ha presionado el mouse
        drawing = True  # En caso de ser verdado se asigna una variable boleana
        ix, iy = x, y  # Almacenamos la poscion incial en las variales
        count = 1
        img[y-r:y+r, x-r:x+r, 2] = (img1[y-r:y+r, x-r:x+r, 0])
        img[y-r:y+r, x-r:x+r, 1] = (img1[y-r:y+r, x-r:x+r, 2])
        img[y-r:y+r, x-r:x+r, 0] = (img1[y-r:y+r, x-r:x+r, 1])
        if count==4:
            count=1
            
            
        """ elif count ==4:
            img[0:y, 0:x, 0] = (img1[0:y, 0:x, 0])
        elif count ==5:
            img[0:y, 0:x, 2] = (img1[0:y, 0:x, 2])
        elif count ==6:
            img[0:y, 0:x, 1] = (img1[0:y, 0:x, 1])
            count=0
            print(count) """

    elif event == cv2.EVENT_MOUSEMOVE:  # Cuando se mueva el moue
        if drawing == True:  # Si se verdadera la condicion de dibujo
            if mode == True: # Si se verdadera la condicion de modo
                #img[0:height, 0:width] = (img1[0:height, 0:width])
                cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 256), thickness=2,)  # Comando para dibujar un rectangulo
            else:
                #cv2.circle(img, (x, y), 8, (0, 0, 255), -1)  # Comando para dibujar un circulo
                if count == 1:
                    #img[0:y, 0:x, 2] = (img1[0:y, 0:x, 0])
                    #img[0:y, 0:x, 0] = (img1[0:y, 0:x, 2])
                    img[y-r:y+r, x-r:x+r, 2] = (img1[y-r:y+r, x-r:x+r, 0])
                    img[y-r:y+r, x-r:x+r, 1] = (img1[y-r:y+r, x-r:x+r, 2])
                    img[y-r:y+r, x-r:x+r, 0] = (img1[y-r:y+r, x-r:x+r, 1])
                    
                elif count ==2:
                    #img[0:y, 0:x, 1] = (img1[0:y, 0:x, 2]) 
                    #img[0:y, 0:x, 2] = (img1[0:y, 0:x, 1])
                    img[y-r:y+r, x-r:x+r, 2] = (img1[y-r:y+r, x-r:x+r, 1])
                    img[y-r:y+r, x-r:x+r, 1] = (img1[y-r:y+r, x-r:x+r, 0])
                    img[y-r:y+r, x-r:x+r, 0] = (img1[y-r:y+r, x-r:x+r, 2])
                    

                elif count ==3: 
                    #img[0:y, 0:x, 0] = (img1[0:y, 0:x, 1]) 
                    #img[0:y, 0:x, 1] = (img1[0:y, 0:x, 0])

                    img[y-r:y+r, x-r:x+r, 2] = (img1[y-r:y+r, x-r:x+r, 2])
                    img[y-r:y+r, x-r:x+r, 1] = (img1[y-r:y+r, x-r:x+r, 1])
                    img[y-r:y+r, x-r:x+r, 0] = (img1[y-r:y+r, x-r:x+r, 0])

               
            

    elif event == cv2.EVENT_LBUTTONUP:  # Cuando se levante el boton
        drawing = False  # Que ya no dibuje
        if mode == True:

            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), thickness=2)
           
        #else:
            #cv2.circle(img, (x, y), 8, (0, 0, 255), 0)
        #print(ix, iy, x, y)





cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)  # Muestro las imagenes

while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(ix, iy)
        # Se esperan 30ms para el cierre de la ventana o hasta que el usuario precione la tecla q
        break

cv2.destroyAllWindows()
