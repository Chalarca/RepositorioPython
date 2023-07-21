import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

# Translacion
img = cv2.imread('Vision Artificial\Archivos\Yian_KutKu.jpg', 0)
rows, cols = img.shape
translation = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, translation, (cols, rows))

cv2.imshow('Imagen original', img)
cv2.imshow('Translacion', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Escalado
scaling = cv2.resize(img, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)

cv2.imshow('Imagen original', img)
cv2.imshow('Escalado 1', scaling)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OR

height, width = img.shape[:2]
res = cv2.resize(img, (50, 520), interpolation=cv2.INTER_CUBIC)

cv2.imshow('Escalado 1', scaling)
cv2.imshow('Escalado 2', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotacion
height, width = img.shape[:2]
rotation = cv2.getRotationMatrix2D((height, width), 45, 1)
dst = cv2.warpAffine(img, rotation, (cols*2, rows*2))

cv2.imshow('Rotacion', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Shearing
transM = np.float32([[1, math.tan(0.30), 0], [0, 1, 0]])
dst = cv2.warpAffine(img, transM, (cols*2, rows*2), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

cv2.imshow('Shearing', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


transM2 = np.float32([[1, -math.tan(0.30), 0], [0, 1, 0]])
rotation = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
#dst = cv2.warpAffine(img, rotation, (cols*2, rows*2))
dst2 = cv2.warpAffine(dst, transM2, (cols*2, rows*2), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

cv2.imshow('Shearing2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Transformacion afine
img = cv2.imread('grid.jpg')
rows, cols, ch = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
afineTrans = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, afineTrans, (cols, rows))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

# Fuente: Documentacion OpenCV
