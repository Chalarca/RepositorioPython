import os
import cv2
from os import mkdir
import tkinter as tk
from tkinter import *
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import *
import time
import serial

##################################################
#ser = serial.Serial('COM3', 9600) # puesto serial
##################################################

def registro_facial():
    global direc
    n=0
    usuario_num = str(dato_numero1.get())    
    direc =str("Ruta"+usuario_num) # direccion de la carpeta donde se almacena cada usuario sengo su documento
    mkdir(direc)  # creamos la carpeta
    archi1=open(direc+'/datos.txt',"w") # creamos el archivo txt
    archi1.write(str(dato_nombre1.get())+"\n")  # guardamos el nombre y cambiamos de renglon
    archi1.write(str(dato_apellido1.get())+"\n") # guardamos apellido
    archi1.write(str(dato_numero1.get())+"\n")  # guardamos el documento estos 3 se pueden hacer en una sola linea de codigo


    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture("video.mp4")
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5) as face_detection:
        while True:
            
            ret, frame = cap.read()
            height, width, _ = frame.shape
            if ret == False:
                break
            #frame = imutils.resize(frame, width=720)
            #frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
            results = face_detection.process(frame_rgb)
            if results.detections is not None:
                n=n+1
                for detection in results.detections:
                    
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)
                    try:
                        ajs=10
                        cv2.rectangle(frame, (xmin-ajs, ymin-ajs-5), (xmin + w+ajs, ymin + h+ajs), (0, 255, 0), 2)
                        #mp_drawing.draw_detection(frame, detection,mp_drawing.DrawingSpec(color=(0, 255, 255), circle_radius=2))
                        imgp_rec=frame[ymin-ajs:ymin-ajs + h,xmin:xmin + w]
                        imgp_rec = cv2.resize(imgp_rec, (240,240), interpolation = cv2.INTER_CUBIC)
                        if n%2==0:
                            cv2.imwrite(direc+"/img"+str(n/2)+".jpg", imgp_rec)
                        cv2.imshow("Frame2", imgp_rec)
                    except:
                        0
                if (n >= 520*3): 
                    break
            ###########################################################################################################################
            #Poner los If de los led con el serial aca
            ###########################################################################################################################
            cont = os.listdir(direc)
            if len(cont) == 2: # envia a al arduino
                #ser.write(num.encode())
                print('a')
                time.sleep(0.1)
                
            if len(cont) == 10: # envia b al arduino
                #ser.write(num.encode())
                print('b')
                time.sleep(0.1)
            
            if len(cont) == 20: # envia c al arduino
                #ser.write(num.encode())
                print('c')
                time.sleep(0.1)
            
            if len(cont) == 30: # envia d al arduino
                #ser.write(num.encode())
                print('d')
                time.sleep(0.1)
                
            if len(cont) >= 40: # cierra la camara cuando se cumpla el nuemro de archivos
                print('fin')
                time.sleep(0.1)
                #ser.write(num.encode())
                cap.release()   
                cv2.destroyAllWindows()
                break
            
        
            texto = "mira hacia la luz"
            cv2.putText(frame,texto, 
            (120,30),cv2.FONT_HERSHEY_TRIPLEX,
            1,(221,0,0),1)
            cv2.imshow("Frame", frame)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def activar(*args):
    
    text1 = dato_nombre1.get().lower()
    text2 = dato_apellido1.get().lower()
    text3 = dato_numero1.get().lower()
    
    if text1 != '' :
        if text2 != '':
            if text3 != '':
                boton.configure(state = NORMAL)
       
    else : boton.configure(state = DISABLED)
    
#contenedor datos de usuario 
    
global nombre1      #Globalizamos la variable para usarla en las funciones
global dato_nombre1
global apellido1
global dato_apellido1
global numero1
global dato_numero1
global boton
    
ventana = Tk()
ventana.geometry("300x250")  #Asignamos el tamaño a la ventana 
ventana.title("bot Industries")  #Asignamos el titulo a la ventana
#icono = PhotoImage(file ="D:\UNIVERSIDAD\Diseño_Mecatronico_2\Comunicacion_Serial\log.png") # ubicacion del icono
#ventana.iconphoto(True, icono)  # activo el icono

Label(text = "datos de registro",
        bg = "white", 
        width = "300",
        height = "2", 
        font = ("Verdana", 13)).pack()  #caracteristicas de la ventana

nombre1 = StringVar()   #creo las variables asignando un tipo de dato
apellido1 = StringVar()
numero1 = IntVar(value= '')
   
# datos de formulario
    
Label(ventana, text = "primer nombres * ").pack()    
dato_nombre1 = Entry(ventana, textvariable = nombre1)
dato_nombre1.pack()
    
    
Label(ventana, text = "primer pellidos * ").pack()
dato_apellido1 = Entry(ventana, textvariable = apellido1)
dato_apellido1.pack()
   
Label(ventana, text = "numero de documento * ").pack()
dato_numero1 = Entry(ventana, textvariable = numero1)
dato_numero1.pack()
    
nombre1.trace_add("write", activar)
apellido1.trace_add("write", activar)
numero1.trace_add("write", activar)
    
Label(text = "").pack() #Creamos el espacio entre el label y el boton
boton = tk.Button(text = "siguiente", 
                height = "2", 
                width = "20", 
                command = registro_facial, 
                #font = 'bold 10',
                state=tk.DISABLED) #caracteristicas del boton que debe tener un mejor nombre

boton.pack()    

ventana.mainloop()