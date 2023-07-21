import numpy as np
import cv2
import mediapipe as mp
import pickle

clasif = pickle.load(open("Vision Artificial\Proyecto Final\Vocales_Gestos.pkl", 'rb'))

contorno=mp.solutions.drawing_utils
mano=mp.solutions.hands
ciclo=0
video = cv2.VideoCapture(0)
posiciones=[]
comprobacion=""
prueba="Nan"
letra=0
Palabra=[]
Num_letras=-1
frase=""
with mano.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=0.1) as hands:

    while True:
        ret, frame = video.read()
        copia=frame
        alto,ancho=frame.shape[:2]
        if ret == False: 
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        image_rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado=hands.process(image_rgb)

        if resultado.multi_hand_landmarks is not None:    
    
            for hand_landmarks in resultado.multi_hand_landmarks:
                0
                contorno.draw_landmarks(frame, hand_landmarks, mano.HAND_CONNECTIONS)

            if ciclo>=3:

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

                if comprobacion==prueba:
                    letra=letra+1
                    if letra>=4:
                        Palabra.append(comprobacion[0])
                        letra=0
                        
                        Num_letras=Num_letras+1
                        frase=frase+Palabra[-1]
                        #print(frase)
                        #print(Palabra)
                        
                else:
                    letra=0
                comprobacion=prueba

                posiciones=[[Pulgar,Indice,Fuck,Anular,Meñique]]
                prueba=clasif.predict(posiciones)


                ciclo=0
            ciclo=ciclo+1
        cv2.putText(frame, frase,(10,alto-20),2,1,(0,255,0), 2)

        cv2.imshow('Frame',frame)
        if cv2.waitKey(30) & 0xFF == ord ('q'):
            break
        
video.release()
