import cv2

img = cv2.imread('25.JPG')
conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('Imagen original', img)  # Mostramos la imagen segementada
cv2.imshow('Imagen convertida', conv)  # Mostramos la imagen segementada
cv2.waitKey(0)
cv2.destroyAllWindows()