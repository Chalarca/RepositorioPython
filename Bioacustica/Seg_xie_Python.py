import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import savgol_filter
from scipy.io import wavfile

def without_subband_mode_intensities(I1):
    M, N = np.shape(I1)
    I2 = np.zeros((M, N), np.double)
    mode1 = np.zeros((1, M), np.double)

    nf = 0
    for nf in range(0, M):
        thisi = I1[nf, :]
        thisi[thisi == np.inf] = np.nan

        maxi = np.max(np.real(thisi[:]))

        mini = np.min(np.real(thisi[:]))

        threshi = np.abs((mini-maxi)/2)

        hvec = np.arange(np.min(np.real(thisi[:])), np.max(np.real(thisi[:])))
        if np.size(hvec) == 1:
            hvec = np.expand_dims(np.linspace(mini, maxi, 2), axis=0)

        histii = np.real(plt.hist(thisi[:], hvec))
        histi = histii[0]

        loc = np.argmax(histi[:])

        mode1_tmp = hvec[loc]
        mode1[0, nf] = mode1_tmp

    # Filtro de promedio Movil
    mode2 = savgol_filter(mode1, 11, 1)
    mode2 = np.transpose(mode2)

    for nf in range(0, M):
        I2[nf, :] = I1[nf, :]-mode2[nf]
    return I2

def findeccentricity (ellipse):
    secelip_1=ellipse[1][0]
    secelip_2=ellipse[1][1]
    if secelip_1>secelip_2:
        elip_a=secelip_1
        elip_b=secelip_2
    else:
        elip_a=secelip_2
        elip_b=secelip_1

    elip_c=np.sqrt((elip_a**2)-(elip_b**2))
    eccentricity=elip_c/elip_a
    return eccentricity

def seg_xie(intensity,specgram_time,specgram_frecuency):
    """Realiza en analisis de los elementos de mayor intensidad en el espectrograma para encontrar
    el tiempo y frecuencia maxima y minima de los elementos mas representativos del audio seleccionado

    Args:
        intensity (array): la variable spectrum, salida de la funsion specgram de matplotlib
        es una arreglo 2D que indica las intensidades sonoras del audio analizado.

        specgram_time (array): Es un arreglo de una dimension que indica el rango de tiempo que
        ocupa cada pixel en el espectrograma, es la salida "t" de la funcion specgram.

        specgram_frecuency (array): es un arreglo 1D que indica el rango de frecuencias que 
        ocupa cada pixel en el spectrograma, es la salida "f" de la funcion specgram.


    Returns:
        segm_xie (array): Arreglo que pose el tiempo y frecuencia minima y maxima de cada
        elemento encontrado. 
        Ejemplo: [tiempo_inicial,tiempo_final,frecuencia_inicial,frecuencia_final]

        segmentos_nor (array): Arreglo que pose la informacion de segm_xie, pero como
        posicion en el arreglo 2 otorganfo el punto inicial y el ancho y alto del elemento. 
        Ejemplo: [posicion_x,posicion_y,ancho,alto]
    """
    
    specgram_time=np.expand_dims(specgram_time,axis=0)
    specgram_frecuency=np.expand_dims(specgram_frecuency,axis=1)
    intensity=intensity[1:,:]
    spectgram_intensity=20*(np.log10(np.abs(intensity)))#funcion para pasar a desibelios. 
    gauss_intensity = cv2.GaussianBlur(spectgram_intensity,(13,13),sigmaX=2,sigmaY=5)#se utiliza un filtro gausiano. 

    with_suband=without_subband_mode_intensities(gauss_intensity)

    with_suband = with_suband * (with_suband>=0)

    cv2.imwrite("Anda2.png",with_suband) #guardo la imagen ya que no se puede manipular directamente.

    with_suband=cv2.imread("Anda2.png",0) #la abro para pocerder con el programa, debo buscar una mejor solucion.
    #with_suband=np.abs(with_suband)
    _,wsub_binarized = cv2.threshold(with_suband, 0, 255,type =cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    wsub_binarized=np.flipud(wsub_binarized) #se binariza con un filtro adaptativo y se invierte

    rectang_Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(7, 9)) #creo el kernel rectangular para la operacion de opening
    morf_opening = cv2.morphologyEx(wsub_binarized, cv2.MORPH_OPEN, rectang_Kernel, iterations=1)
    cuad_Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(6, 6)) #kernel para el Closening
    morf_close = cv2.morphologyEx(morf_opening, cv2.MORPH_CLOSE, cuad_Kernel, iterations=1)

    spectgram_contours, hierarchy = cv2.findContours(morf_close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #encuentro todos los grupos de pixeles blancos unidos
    spectgram_estructures=[]

    #filtrando los contornos de acuerdo a su tamaño, area en bounding box y morfologia de la ecentricidad(si es circulo o linea)
    for cnt in spectgram_contours:
        x,y,w,h = cv2.boundingRect(cnt) #encontrando la bounding box del elemento
        area=cv2.contourArea(cnt) #encontrando su area
        exent=area/(w*h)
        try:
            if area>200 and area<40000 and exent>0.3:
                ellipse = cv2.fitEllipse(cnt) #me da los elemetos que componen una elipse
                eccentricity=findeccentricity(ellipse) #uso una funcion que cree para encontrar la exentricidad del elemento

                if eccentricity>0.5:
                    spectgram_estructures.append(cnt)  #guardo los elementos que pasan la condicion
            else:
                continue
        except:
            0      
    segment=[] #Arreglo que pose el tiempo y frecuencia minima y maxima.
    segmentos_nor=[] #pose lo mismo que el anterior pero da la posicion en pixeles
    for element in spectgram_estructures:
        timeI,frecma,duration,magfrec=cv2.boundingRect(element)
        posicion=[int(timeI),int(frecma),int(duration),int(magfrec)]
        segment.append([float(specgram_time[:,(posicion[0]+1)]),float(specgram_time[:,(posicion[0]+posicion[2])]),
        float(specgram_frecuency[(posicion[1]+1),:]),float(specgram_frecuency[(posicion[1]+posicion[3]),:])])
        segmentos_nor.append([posicion[0],posicion[1],posicion[2],posicion[3]])
    segm_xie=np.array(segment)
    np.array(segmentos_nor)

    return segm_xie,segmentos_nor

fs, senial = wavfile.read(r"D:\\Users\ACER\Desktop\Trabajo Investigacion\Aureas Mono especies\Audio2\JAGUAS259_20121116_081604.wav")

p,f,t,d=plt.specgram(senial[:,1],Fs=fs,NFFT=2048,noverlap=1550)

segm_xie,segmentos_mor=seg_xie(p,t,f)

print(np.shape(segm_xie),np.shape(segmentos_mor))