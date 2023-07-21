import cv2
import numpy as np
i=0
ListaPxblan=[]
Tolerancia_Deteccion=50 #% de pixeles que tolera respecto a la base antes de levantar la alarma, entre mas bajo mas extricto. 

cap = cv2.VideoCapture(0)

_,base=cap.read()
y, x = base.shape[:2]
fondo = np.zeros(base.shape[:2], np.uint8)
bg=fondo

while True:
    if i==20:
        _,base=cap.read()

        fondo = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        bg=cv2.GaussianBlur(fondo, (21,21), sigmaX=0, sigmaY=0)

    if i >15:
        _ , frame = cap.read()
        Gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Gb=cv2.GaussianBlur(Gframe, (21,21), sigmaX=0, sigmaY=0)
        
        intruso=cv2.bitwise_xor(bg,Gb)
        _,s=cv2.threshold(intruso, 40, 255, cv2.THRESH_BINARY)

        if i>20:
            PixBlan=np.sum(s)/255
            ListaPxblan.append(PixBlan)
        if i == 80:
            limite=np.max(np.array(ListaPxblan))
            print("esto es el mayor",limite)

        if i>101:
            if limite*(Tolerancia_Deteccion/100 +1)< PixBlan: 
                cv2.putText(Gframe, text='Intruso Detectado', org=(30,y-50),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255,0,0),
                thickness=4, lineType=cv2.LINE_8)
        
            else:
                cv2.putText(Gframe, text='OK', org=(30,y-50),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255,0,0),
                thickness=4, lineType=cv2.LINE_8)
        
        
        cv2.imshow("video",Gframe)
        cv2.imshow("wise",s)
        

    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
