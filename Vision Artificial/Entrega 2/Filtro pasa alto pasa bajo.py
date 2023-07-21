import numpy as np
import cv2

img1 = cv2.imread('Vision Artificial\Archivos\Aleatron.jpg')
height, width = img1.shape[:2] 
imgFpb = np.zeros((height, width), np.uint8)

kern2= np.array(
    [[2, -2, 2, -1, -3], 
    [-2, -2, 1,  3, -1],
    [ 2, -1, 4, -1,  2],
    [-1,  3, 1, -2, -2],
    [-3, -1, 2, -2,  2]])#Filtro pasa altos, restalta los bordes
kern3= np.array([
    [  0,    0,  1/19, 1/19, 1/19], 
    [  0,  1/19, 1/19, 1/19, 1/19],
    [1/19, 1/19, 1/19, 1/19, 1/19],
    [1/19, 1/19, 1/19, 1/19,   0 ],
    [1/19, 1/19, 1/19,   0,    0 ]])#filtro pasa bajos, difumina los bordes. 


imgFpb = cv2.filter2D(img1, ddepth=-1, kernel=kern3, anchor=(-1, -1))
ImgPasaA = cv2.filter2D(img1, ddepth=-1, kernel=kern2, anchor=(-1, -1))
cv2.imshow('Resultados', np.hstack([img1, imgFpb,ImgPasaA]))
cv2.waitKey(0) 
cv2.destroyAllWindows() 
