import cv2
import numpy as np

captura=cv2.VideoCapture(0)


while True:
  _,frame = captura.read()
  #src = cv2.medianBlur(frame, 5)
  src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  edged=cv2.Canny(src,50,150)
  kernel = np.ones((5,5),np.uint8)
  #kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  dilate=cv2.dilate(edged,kernel,iterations=1)
  
  total=0
  for c in dilate:

    area=cv2.contourArea(c)
     #print "area",area

    if area>1700:
    #aproximacion de contorno
      peri=cv2.arcLength(c,True) #Perimetro
      approx=cv2.approxPolyDP(c,0.02*peri,True)
                        #Si la aproximacion tiene 4 vertices correspondera a un rectangulo (Libro)
      if len(approx)==4:
         cv2.drawContours(frame,[approx],-1,(0,255,0),3,cv2.LINE_AA)
         total+=1

        #5.Poner texto en imagen
  letrero= 'Objetos: '+ str(total)
  cv2.putText(frame,letrero,(10,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
  
  cv2.imshow('Frame',dilate)
 
  if cv2.waitKey(30) & 0xFF == ord ('q'):
       break 
captura.release()  
cv2.destroyAllWindows()   
     
    
    
    
    
