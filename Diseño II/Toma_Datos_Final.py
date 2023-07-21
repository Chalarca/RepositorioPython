import tkinter as tk
from tkinter import *
import cv2
import os
from os import mkdir
import serial
from tkinter import messagebox
import ctypes
from tkinter import font, ttk
import mediapipe as mp

#ser = serial.Serial('COM3', 9600) # puesto serial


# abre camara y guarda datos del reconocimiento facial

def registro_facial():
    
    ventana2 = cv2.VideoCapture(0)                #Elegimos la camara con la que vamos a hacer la deteccion
    global n 
    global direc
    n = 0
    num = 0
    num1 = '1'
    num2 = '0'
    texto = "Mirar la luz arriba"
    datapath='D:\Proyectos Visual Studio\Python-Jupyther\Diseño II'
    os.chdir(datapath)
    usuario_num = str(dato_numero1.get())    
    direc =str('imageData'+usuario_num) # direccion de la carpeta donde se almacena cada usuario sengo su documento
    mkdir(direc)                                  #creamos la carpeta
    archi1=open(direc+'/datos.txt',"w")           #creamos el archivo txt
    archi1.write(str(dato_nombre1.get())+"\n")    #guardamos el nombre y cambiamos de renglon
    archi1.write(str(dato_apellido1.get())+"\n")  #guardamos apellido
    archi1.write(str(dato_numero1.get())+"\n") 
    user32 = ctypes.windll.user32
    ancho, alto = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)   #guardamos el documento estos 3 se pueden hacer en una sola linea de codigo
    print('dire',direc)
    
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while(True):
            ret,frame = ventana2.read()
            height, width, _ = frame.shape
            if ret == False:
                break
            (a, an) = frame.shape[:2] 
            ancho2 = int((ancho - an)/2)
            alto2 = int((alto - a)/2)              # leemos el video

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
            results = face_detection.process(frame_rgb)
            if results.detections is not None:
                cuantia=2
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
                        if n%cuantia==0:
                            cv2.imwrite(direc+"/img"+str(n/cuantia)+".jpg", imgp_rec)
                            
                        cv2.imshow("Frame2", imgp_rec)
                    except:
                        0
                if (n >= 500*cuantia): 
                    break
            #if n >= 10:
            #    cv2.imwrite(direc+'/img'+str(n)+'.jpg', frame)
            #    num = num + 1
            #n = n + 1

            num=n

            #cv2.putText(frame,"Manten tu cara dentro del ovalo", 
            #    (50,a-30),cv2.FONT_HERSHEY_TRIPLEX,
            #    1,(255,0,0),1)
            

            #La Elipce
            #a = int(a/2)
            #an = int(an/2)
            #cv2.ellipse(frame, (an, a), 
            #    (100, 140),0, 0,
            #    360, (255,0,0),3)   
            ###    
            
            cv2.putText(frame,texto, 
                (135,30),cv2.FONT_HERSHEY_TRIPLEX,
                1,(255,0,0),1)
            #cv2.putText()
            cv2.imshow('Registro Facial',frame)       # mostramos el video en pantalla
            cv2.moveWindow('Registro Facial', ancho2, alto2)
            ventana.withdraw() # ocultar ventana
            #cv2.setWindowProperty('Registro Facial', cv2.WND_PROP_TOPMOST, 1)
            if num == 1*cuantia: # envia a al arduino
                #ser.write(num1.encode())
                print('a')
                messagebox.showinfo(message="Mira la luz de color rojo, arriba\nPresiona espacio o enter para continuar",
                                    title="Advertencia")
            if num == 125*cuantia: # envia a al arduino
                #ser.write(num1.encode())
                texto = "Mira la luz a la Derecha"
                print('b')
                messagebox.showinfo(message="Mira la luz de color verde, a la derecha\nPresiona espacio o enter para continuar",
                                    title="Advertencia")
            if num == 250*cuantia: # envia a al arduino
                #ser.write(num1.encode())
                print('c')
                texto = "Mira la luz Abajo"
                messagebox.showinfo(message="Mira la luz de color verde, abajo\nPresiona espacio o enter para continuar",
                                    title="Advertencia")
            if num == 375*cuantia: # envia a al arduino
                #ser.write(num1.encode())
                print('d')
                texto = "Mira la luz a la Izquierda"
                messagebox.showinfo(message="Mira la luz de color verde, a la izquierda\nPresiona espacio o enter para continuar", 
                                    title="advertencia")
            if num >= 500*cuantia:      
                #ser.write(num2.encode()) 
                messagebox.showinfo(message="gracias por realizar tu registro\nPresiona espacio o enter para continuar",
                                    title="advertencia")
                break
            cv2.waitKey(30)
    
        ventana.deiconify()    
        ventana2.release()                            #Cerramos
        cv2.destroyAllWindows()  
        dato_nombre1.delete(0, END)  
        dato_apellido1.delete(0, END)
        dato_numero1.delete(0, END)
        
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
    
global nombre1                    #Globalizamos la variable para usarla en las funciones
global dato_nombre1
global apellido1
global dato_apellido1
global numero1
global dato_numero1
global boton
    
ventana = tk.Tk()
ventana.deiconify() # mostramos la ventana
ventana.geometry("300x340")    # Asignamos el tamaño a la ventana 
ventana.title("Bot Industries") #Asignamos el titulo a la ventana
ventana.config(bg="#2E2D2C") # color fondo de la ventana
# activo el icono
ventana.eval('tk::PlaceWindow . center')          # activo el icono


Label(text = "Datos De Registro",
        bg = "#1F1E1D", # color  fondo
        fg='#ad4545',  # color letra
        width = "300",
        height = "2", 
        font = font.Font(family="Times", size=14),).pack()     #caracteristicas de la ventana

nombre1 = StringVar()                      #creo las variables asignando un tipo de dato
apellido1 = StringVar()
numero1 = IntVar(value= '')
    
    # datos de formulario
        
Label(ventana, 
      bg="#2E2D2C", # color fondo
      fg='#ad4545', # color de la letra
      font = font.Font(family="Times", size=14),
      text = "\nPrimer nombre * ").pack()    
dato_nombre1 = Entry(ventana, 
                     textvariable = nombre1,
                     bg='white',
                     fg='black',
                     font = font.Font(family="Times", size=14) )
dato_nombre1.pack()
        
        
Label(ventana, 
      bg = "#2E2D2C", 
      fg='#ad4545',
      font = font.Font(family="Times", size=14),
      text = "Primer apellido * ").pack()
dato_apellido1 = Entry(ventana,
                       bg = 'white',
                       fg='black',
                       textvariable = apellido1,
                       font = font.Font(family="Times", size=14))
dato_apellido1.pack()
    
Label(ventana, 
      bg = "#2E2D2C", 
      fg='#ad4545',
      font = font.Font(family="Times", size=14),
      text = "Número de documento *").pack()
Label(ventana, 
      bg = "#2E2D2C", 
      fg='#ad4545',
      font = font.Font(family="Times", size=10),
      text = "(Sin espacios o signos)").pack()
dato_numero1 = Entry(ventana,
                     bg = 'white',
                     fg='black',
                     textvariable = numero1,
                     font = font.Font(family="Times", size=14))
dato_numero1.pack()

nombre1.trace_add("write", activar)
apellido1.trace_add("write", activar)
numero1.trace_add("write", activar)

Label(text = "", bg="#2E2D2C").pack()                 #Creamos el espacio entre el label y el boton
boton = tk.Button(text = "Siguiente", 
                bg = "white",
                fg = "#ad4545",
                height = "2", 
                width = "20", 
                command = registro_facial, 
                font = font.Font(family="Times", size=14),
                state=tk.DISABLED) # caracteristicas del boton que debe tener un mejor nombre

boton.pack() 
ventana.mainloop()  