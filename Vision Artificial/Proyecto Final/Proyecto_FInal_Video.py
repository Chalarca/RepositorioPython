import numpy as np
import cv2
import mediapipe as mp
import pandas as pd

contorno=mp.solutions.drawing_utils
mano=mp.solutions.hands
ciclo=0
video = cv2.VideoCapture(0)
posiciones=[]
Etiqueta="U"
with mano.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=0.1) as hands:

    while True:
        ret, frame = video.read()
        copia=frame
        if ret == False: 
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        image_rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado=hands.process(image_rgb)


        #gray=cv2.cvtColor(copia,cv2.COLOR_BGR2GRAY)
        #_,th=threshold(gray,100,255,cv2.THRESH_BINARY)
        #img,contorno,jerarquia=cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #dibujocontorno=cv2.drawContours(frame,contorno,-1,(0,55,0),3)


        if resultado.multi_hand_landmarks is not None:    
            #print(resultado.multi_hand_landmarks)
        
    
            for hand_landmarks in resultado.multi_hand_landmarks:
                contorno.draw_landmarks(frame, hand_landmarks, mano.HAND_CONNECTIONS)
                #print(int(hand_landmarks.landmark[0].x*width))
                
            if ciclo>=1:

                x1 = int(hand_landmarks.landmark[4].x * width)
                y1 = int(hand_landmarks.landmark[4].y * height)
                x2 = int(hand_landmarks.landmark[8].x * width)
                y2 = int(hand_landmarks.landmark[8].y * height)
                x3 = int(hand_landmarks.landmark[12].x * width)
                y3 = int(hand_landmarks.landmark[12].y * height)
                x4 = int(hand_landmarks.landmark[16].x * width)
                y4 = int(hand_landmarks.landmark[16].y * height)
                x5 = int(hand_landmarks.landmark[20].x * width)
                y5 = int(hand_landmarks.landmark[20].y * height)
                x6 = int(hand_landmarks.landmark[0].x * width)
                y6 = int(hand_landmarks.landmark[0].y * height)
                x7 = int(hand_landmarks.landmark[9].x * width)
                y7 = int(hand_landmarks.landmark[9].y * height)

                Referencia=np.sqrt((x7-x6)**2+(y7-y6)**2)
                Pulgar=np.sqrt((x1-x6)**2+(y1-y6)**2)/Referencia
                Indice=np.sqrt((x2-x6)**2+(y2-y6)**2)/Referencia
                Fuck=np.sqrt((x3-x6)**2+(y3-y6)**2)/Referencia
                Anular=np.sqrt((x4-x6)**2+(y4-y6)**2)/Referencia
                Meñique=np.sqrt((x5-x6)**2+(y5-y6)**2)/Referencia

                posiciones.append({"Pulgar":Pulgar,"Indice":Indice,"Medio":Fuck,"Anular":Anular,"Meñique":Meñique,"Letra":Etiqueta})

                #print(f"Pulgar:",Pulgar,"Indice:",Indice, "Fuck:",Fuck, "Anular:",Anular,"Meñique:",Meñique,"Referencia:",Referencia)
                #print(f"P",x6,y6)
                #print(f"P",x1,y1,"A",x2,y2,"M",x3,y3,"R",x4,y4,"Ñ",x5,y5,"C0",x6,y6)
                ciclo=0
            ciclo=ciclo+1
             
        cv2.imshow('Frame',frame)
        if cv2.waitKey(30) & 0xFF == ord ('q'):
            break
        
video.release()
#print(posiciones)   
Datos=pd.DataFrame(posiciones)
Datos.drop(index=0)
Datos.to_excel("Vision Artificial\Proyecto Final\Datos_Exesasas.xlsx",sheet_name="Letra",engine="openpyxl")
print(Datos)