from tkinter.tix import Tree
import cv2
import numpy as np

drawing = False  # Verdadero si el mouse es presionado
mode = True  # Si es verdadero, dibuje un rectangulo, de lo contrario un circulo
ix, iy = -1, -1

#  funcion de llamado del mouse
def draw(event, x, y, flags, param): # Se declara la funcion
    global ix, iy, drawing, mode  # Defino unas variables globales

    if event == cv2.EVENT_LBUTTONDOWN:  # Se pregunta si se ha presionado el mouse
        drawing = True  # En caso de ser verdado se asigna una variable boleana
        ix, iy = x, y  # Almacenamos la poscion incial en las variales

    elif event == cv2.EVENT_MOUSEMOVE:  # Cuando se mueva el moue
        if drawing == True:  # Si se verdadera la condicion de dibujo
            if mode == True: # Si se verdadera la condicion de modo
                img[0:height, 0:width] = (img1[0:height, 0:width])
                cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), thickness=2,)  # Comando para dibujar un rectangulo
            else:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)  # Comando para dibujar un circulo
            

    elif event == cv2.EVENT_LBUTTONUP:  # Cuando se levante el boton
        drawing = False  # Que ya no dibuje
        if mode == True:

            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), thickness=2)
           
        else:
            cv2.circle(img, (x, y), 1, (0, 0, 255), 0)
        #print(ix, iy, x, y)

img1 = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg')  # Leemos la imagen
height, width = img1.shape[:2]
img = np.zeros((height, width, 3), np.uint8)  # Creo una imagen vacia
img[0:height, 0:width] = (img1[0:height, 0:width])

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)  # Muestro las imagenes

while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(ix, iy)
        # Se esperan 30ms para el cierre de la ventana o hasta que el usuario precione la tecla q
        break

cv2.destroyAllWindows()

# Fuente: Documentación OpenCV