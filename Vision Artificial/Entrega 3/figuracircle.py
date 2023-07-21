
import cv2
import numpy as np

video = cv2.VideoCapture(0)



while True:
    ret,frame = video.read()
    if not ret : break
    
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    salida = frame.copy()
    
    blur = cv2.GaussianBlur(grayframe, (17,17),0)
    can = cv2.Canny(blur,50,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))  # Definir núcleo rectangular
    dilation = cv2.dilate(can, kernel, iterations=1)
    row = dilation.shape[0]
    circles = cv2.HoughCircles(dilation, cv2.HOUGH_GRADIENT,2,600,
                               param1 =50, 
                               param2 =30,
                               minRadius=50, 
                               maxRadius=100)
    
    circles = np.uint16(np.around(circles))
   
    for i in circles[0,:]:
        centrox = i[0]
        centroy = i[1]
        radio = i[2]
        cv2.circle(salida,(centrox,centroy),radio,(255,0,0),3)
    
    
    cv2.imshow("circles", salida)                
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()    