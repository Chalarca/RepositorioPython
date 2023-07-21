import cv2
import numpy as np

cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w = frame.shape[:2]
a=80
c=0
barsWidth = []
while 1:

    # Frame by frame capture
    _, frame = cap.read()
    if c == 30:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale conversion
        # gray = cv2.GaussianBlur(gray, (11,11), 0, 0)
        snipped = gray[h//2-a:h//2+a, w//2-(2*a):w//2+(2*a)]

        threshVal, thresh = cv2.threshold(snipped, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        line = np.count_nonzero(thresh, axis=0)>a
        l  = np.unique(line.cumsum()[~line])
        barsWidth = l[1:] - l[:-1]
        c=0
    cv2.rectangle(frame, (w//2-(2*a),h//2-a), (w//2+(2*a)+1,h//2+a+1), (0,0,255))
    cv2.putText(frame, str(barsWidth)[1:-1],(0,h-10),2,0.6,(0,0,255), 2)
    c = c+1
    print(barsWidth)
    
    # thresh[:, line] = 255
    # thresh[:,~line] = 0
    # cv2.imshow('Thresh',thresh)
    
    cv2.imshow('Video',frame) #Original image
    # print(f'Umbral escogido mdiante Otsu: {threshVal}')
    # print(f'Cantidad de barras negras: {len(barsWidth)}')
    # print(f'Grosor de las barras: {barsWidth}')
    print(barsWidth)
    if cv2.waitKey(1) & 0xFF == 27:  #Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()