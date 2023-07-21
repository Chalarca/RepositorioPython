import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Inicio de captura

while(1):
    _, frame = cap.read()  # Almacenamiento del frame
    height, width = frame.shape[:2]  # Captura de las dimensiones
    segImg1 = np.zeros(frame.shape, np.uint8)  # Creacion de la imagen vacia para la segmentacion 1
    segImg2 = np.zeros((height, width), np.uint8)  # Creacion de la imagen vacia para la segmentacion 2
    lower = np.array([0, 0, 0])  # Limite inferior
    upper = np.array([100, 100, 100])  # Limite superior
    mask = cv2.inRange(frame, lower, upper) # Creaion de la mascara con los rangos a segmentar
    segImg1 = cv2.bitwise_and(frame, frame, mask=mask)  # Segmentacion1: operacion de conjuncion logica
    segImg2 = cv2.inRange(frame, lower, upper)  # Segemetacion2: funcion inrange
    cv2.imshow('Original', frame)  # Imagen original
    cv2.imshow('Segmentation1', segImg1)  # Segmentacion1
    cv2.imshow('Segmentation2', segImg2)  # Segmentacion2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Fuente: Documentacion OpenCV
