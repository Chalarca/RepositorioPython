# Jorge Torres Arboleda
# Filtro iterativo para ruido de imagen 4

import cv2
import numpy as np


def SaltPepperFilter(image,*flag, kernelsize=3, show=False):
    
    img = cv2.imread(image,*flag) if isinstance(image, str) else image  
    
    if isinstance(kernelsize, int):
        
        kernelsize += 1 if kernelsize % 2 == 0 else 0
        fwx = (kernelsize-1)//2 #Frame width, grosor del marco
        fwy = fwx
        
    elif len(kernelsize) == 2:
        
        sizeX, sizeY = kernelsize
        sizeX += 1 if sizeX % 2 == 0 else 0
        sizeY += 1 if sizeY % 2 == 0 else 0
        fwx = (sizeX-1)//2 #Frame width left and right
        fwy = (sizeY-1)//2 #Frame width top and bottom
        
    else:
        raise "Error kernel dimension"
        
    def median(img,fwy=2,fwx=2):
        img2 = cv2.copyMakeBorder(img,fwy,fwy,fwx,fwx,cv2.BORDER_REPLICATE)  #Image with frame border replicate
        
        shape = img.shape
        height, width = shape[:2]
        img3 = np.zeros((int(height), int(width), 3), np.uint8) if len(shape)==3 else np.zeros((int(height), int(width)), np.uint8)

        for j in range(fwy,height+fwy):
            for i in range(fwx,width+fwx):
                m = img2[j-fwy:j+fwy+1,i-fwx:i+fwx+1]
                img3[j-fwy,i-fwx] = np.median(m) 
        return img3
    
    # cv2.imshow('Intemediate',median(img,fwy,fwx))
    # cv2.imwrite('E-Intermediate.jpg',median(img,fwy,fwx))
    filtered = median(median(img,fwy,fwx))
   
    if show:
        cv2.imshow('Filtered', filtered)
    return filtered




img = cv2.imread('Vision Artificial\Entrega 2\Imagenes\img4.png',0) # Read image in gray scale
# cv2.imshow('Original', img)
filtered = SaltPepperFilter(img,kernelsize=3, show=False)
cv2.imshow('Imagen Original - Imagen Filtrada', np.hstack([img, filtered]))
# cv2.imwrite('E-Result.jpg', np.hstack([img, filtered]))
cv2.waitKey()
cv2.destroyAllWindows()