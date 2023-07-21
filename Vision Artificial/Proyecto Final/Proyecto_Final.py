import numpy as np
import cv2
import mediapipe as mp

contorno=mp.solutions.drawing_utils
mano=mp.solutions.hands

with mano.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=0.5) as hands:

    image = cv2.imread("Vision Artificial\Archivos\IMG_20141203_095016.jpg")
    height, width, _ = image.shape
    image = cv2.flip(image, 1)
    image_rgb =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resultado=hands.process(image_rgb)

    print("multihands:", resultado.multi_handedness)

    #print("Landmarks", resultado.multi_hand_landmarks)
    if resultado.multi_hand_landmarks is not None:    
    # Dibujando los puntos y las conexiones mediante mp_drawing 
        for hand_landmarks in resultado.multi_hand_landmarks:
            contorno.draw_landmarks(
            image, hand_landmarks, mano.HAND_CONNECTIONS,
            contorno.DrawingSpec(color=(255,255,0), thickness=4, circle_radius=5),
            contorno.DrawingSpec(color=(255,0,255), thickness=4))

            x1 = int(hand_landmarks.landmark[mano.HandLandmark.THUMB_TIP].x * width)
            y1 = int(hand_landmarks.landmark[mano.HandLandmark.THUMB_TIP].y * height)
            x2 = int(hand_landmarks.landmark[mano.HandLandmark.INDEX_FINGER_TIP].x * width)
            y2 = int(hand_landmarks.landmark[mano.HandLandmark.INDEX_FINGER_TIP].y * height)
            x3 = int(hand_landmarks.landmark[mano.HandLandmark.MIDDLE_FINGER_TIP].x * width)
            y3 = int(hand_landmarks.landmark[mano.HandLandmark.MIDDLE_FINGER_TIP].y * height)
            x4 = int(hand_landmarks.landmark[mano.HandLandmark.RING_FINGER_TIP].x * width)
            y4 = int(hand_landmarks.landmark[mano.HandLandmark.RING_FINGER_TIP].y * height)
            x5 = int(hand_landmarks.landmark[mano.HandLandmark.PINKY_TIP].x * width)
            y5 = int(hand_landmarks.landmark[mano.HandLandmark.PINKY_TIP].y * height)

            print(x1,y3)
            print(x2,y2)
            print(x3,y3)
            print(x4,y4)
            print(x5,y5)
          

    image = cv2.flip(image, 1)

    cv2.imshow('Mano', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()