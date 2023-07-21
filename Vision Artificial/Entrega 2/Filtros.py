import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

img1 = cv2.imread('Vision Artificial\Entrega 2\Imagenes\img1.png')
img2 = cv2.imread('Vision Artificial\Entrega 2\Imagenes\img2.png')
img3 = cv2.imread('Vision Artificial\Entrega 2\Imagenes\img3.png')

kern1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kern2 =  np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
""" kern2= np.array(
    [[2, -2, 2, -1, -3], 
    [-2, -2, 1,  3, -1],
    [ 2, -1, 4, -1,  2],
    [-1,  3, 1, -2, -2],
    [-3, -1, 2, -2,  2]]) """#Filtro pasa altos, restalta los bordes
kern3= np.array([
    [  0,    0,  1/19, 1/19, 1/19], 
    [  0,  1/19, 1/19, 1/19, 1/19],
    [1/19, 1/19, 1/19, 1/19, 1/19],
    [1/19, 1/19, 1/19, 1/19,   0 ],
    [1/19, 1/19, 1/19,   0,    0 ]])#filtro pasa bajos, difumina los bordes. 

def Filtros(imgin,kern1,kern2,kern3, Gb, Medb):
    
    img = cv2.cvtColor(imgin, cv2.COLOR_BGR2RGB)
    filt1 = cv2.GaussianBlur(img, (Gb,Gb), sigmaX=0, sigmaY=0)
    filt2 = cv2.medianBlur(img, Medb)
    filt3= cv2.filter2D(img, ddepth=-1, kernel=kern1, anchor=(-1, -1))
    filt4 = cv2.filter2D(img, ddepth=-1, kernel=kern2, anchor=(-1, -1))
    filt5 = cv2.filter2D(img, ddepth=-1, kernel=kern3, anchor=(-1, -1))

    gs = gridspec.GridSpec(2, 3)
    plt.subplot(gs[0, 0]), plt.imshow(img)  
    plt.title('Original')
    plt.subplot(gs[0, 1]), plt.imshow(filt1)
    plt.title('Filtro Gausiano') 
    plt.subplot(gs[0, 2]), plt.imshow(filt2) 
    plt.title('Filtro Mediana')
    plt.subplot(gs[1, 0]), plt.imshow(filt3)
    plt.title('Filtro Sharpen')
    plt.subplot(gs[1, 1]), plt.imshow(filt4)
    plt.title('Filtro Pasa Alto')
    plt.subplot(gs[1, 2]), plt.imshow(filt5)  
    plt.title('Filtro Pasa Bajo')
    plt.show()

Filtros(img1,kern1,kern2,kern3, 7, 3)
Filtros(img2,kern1,kern2,kern3, 7, 3)
Filtros(img3,kern1,kern2,kern3, 7, 3)