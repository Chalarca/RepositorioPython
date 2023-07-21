# coding=utf-8
import cv2
import numpy as np
 
# Lectura de la video camara

#cap = cv2.VideoCapture('http://192.168.1.2:4747/video')
cap = cv2.VideoCapture(0)
while True:
    # Captura cuadro a cuadro
    _, frame = cap.read()
    
    # Conversion de la captura a escala de grises
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('video', gray)
    cv2.imshow('video2', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Se esperan 30ms para el cierre de la ventana o hasta que el usuario precione la tecla q
        break

# Al terminar finalice la captura

cap.release()
cv2.destroyAllWindows()

# Fuente: Documentación OpenCV
