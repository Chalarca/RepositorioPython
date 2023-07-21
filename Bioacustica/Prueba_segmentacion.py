from ctypes import sizeof
#from funciones_aureas import*
from msvcrt import kbhit
import os 
from os.path import isfile, join
from scipy.signal import savgol_filter
from scipy.signal import medfilt2d
import numpy as np
import scipy.io as sio
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import cv2
from scipy.io import wavfile
import numpy.matlib
from scipy import signal
from scipy.fft import fftshift
from Bioacustica_Com1 import time_and_date,segmentacion,seg_xie,fcc5

ruta="D:\\Users\ACER\Desktop\Trabajo Investigacion\Aureas Mono especies\Audios"
canal=1
autosel=0
visualize=0
banda=["min","max"]

fechas,cronologia,audios=time_and_date(ruta)

#####Funcion segmentacion########
# esta se programa dentro de primer for con i = 1
# se asume que el espectrograma genera los datos para s,f,t,p, los cuales son tomados de matlab
#y1 = data['y']  
banda=np.array(banda)

#p=0
segment_data=[]
contador_archivos=-1
nombre_archivo=[]

# Aqui se debe llamar la funcion del espectrograma.

for archivo in audios:
    print(archivo)
    contador_archivos=contador_archivos+1

    fs, senial = wavfile.read(archivo)
    if np.shape(senial.shape[:])[0] <2:
        frecuency,time,intensity=signal.spectrogram(senial,fs=fs,nfft=2048,nperseg=569,noverlap=71)
    else:
        #senial = senial.mean(axis=1)
        #senial = np.squeeze(senial)#Se promedian las 2 bandas para tener una sola, antes simplemente elegia 1 de ellas.
        #frecuency,time,intensity=signal.spectrogram(senial,fs=fs,nfft=2048,nperseg=569,noverlap=71)
        frecuency,time,intensity=signal.spectrogram(senial[:,1],fs=fs,nfft=2048,nperseg=569,noverlap=71)
    segm_xie_band=np.empty((0,4),float)
    segmentos_nor_band=np.empty((0,4),float)
    
    s = np.abs(intensity)
    u, v = np.shape(s)
    #resiz=len(y1[:,canal])/len(s[1,:])
    band_1=1/u      # mirar si se usa para fmin
    band_2=1 
    #intensity,frecuency,time,d=plt.specgram(senial[:,1],Fs=fs,NFFT=2048,noverlap=1550)

    mfband = medfilt2d(s,kernel_size=(5,5))
    selband=np.flip(mfband,axis=0)

    #--------------------------  Xie ----------------------------------------
    if type(banda[1])==np.str_:
        banda_aux=np.array([0,frecuency.max()])
    else:
        0
    segm_xie,segmentos_nor=seg_xie(intensity,time,frecuency)
    print(segm_xie.shape[:])
    for k in range(len(segm_xie[:,1])):
        try:
            ti = np.array(segm_xie[k,0]) #tiempo inicial (X)
            tf =np.array(segm_xie[k,1])   #tiempo final(X+W)
            fi = np.array(segm_xie[k,3])    #frecuencia inicial (Y)
            fff = np.array(segm_xie[k,2])    # frecuencia final (Y+H)
            
            if fi>=banda_aux[0] and fff <= banda_aux[1]:
                segm_xie_band=np.append(segm_xie_band, np.expand_dims(np.array([segm_xie[k,0],segm_xie[k,1],segm_xie[k,2],segm_xie[k,3]]),axis=0), axis=0)
                segmentos_nor_band=np.append(segmentos_nor_band, np.expand_dims(np.array([segmentos_nor[k,0],segmentos_nor[k,1],segmentos_nor[k,2],segmentos_nor[k,3]]),axis=0), axis=0)     
        except:
            0

    segm_xie = segm_xie_band
    segmentos_nor=segmentos_nor_band

    k=0
    for k in range(len(segm_xie[:,1])):
        print(len(segment_data))
        try:
            ti = np.array(segm_xie[k, 0])    # tiempo inicial (X)
            tf =np.array(segm_xie[k,1])      #tiempo final(X+W)
            fi = np.array(segm_xie[k,3])     #frecuencia inicial (Y)
            fff = np.array(segm_xie[k,2])    # frecuencia final (Y+H)
                    
            x=np.array(segmentos_nor[k,0])+1                  #tiempo inicial (X)
            xplusw =segmentos_nor[k,0] + segmentos_nor[k,2]   #Tiempo final(X+W)
            y = segmentos_nor[k,1]+1                          #frecuencia inicial (Y)
            yplush = segmentos_nor[k,1] + segmentos_nor[k,3]  #frecuencia final (Y+H)
            seg = np.array(selband[int(y-1):int(yplush),int(x-1):int(xplusw)])
            nfrec = 4
            div= 4
            nfiltros=14 # se cambia porque con 30 se pierden muchos cantos
            features= fcc5(seg,nfiltros,div,nfrec) #50 caracteristicas FCCs

            fseg,cseg=np.shape(seg)
            seg=( (seg-(np.matlib.repmat((np.min(np.real(seg[:]))),fseg,cseg))) 
                / ((np.matlib.repmat((np.max(np.real(seg[:]))),fseg,cseg))
                -(np.matlib.repmat((np.min(np.real(seg[:]))),fseg,cseg))))

            sum_domin=np.transpose(np.expand_dims(np.sum(seg,1),axis=0)) #cambio frecuencia dominante
                    
            dummy,dom = (np.max(np.real(np.transpose(np.expand_dims(savgol_filter(np.ravel(sum_domin)
                        ,1,0),axis=0))))), np.argmax(savgol_filter(np.ravel(sum_domin),1,0))
            
            dom=((((fi*u/(fs/2))+dom)/u)*fs/2) #frecuencia dominante

            dfcc = np.diff(features,1)
            dfcc2 = np.diff(features,2)
            cf = np.cov(features)
            ff = []
            for r in range(len(features[:,0])-1): 
                ff = np.append(ff,np.diag(cf),axis=0)

            #transforma la matriz en un vector tipo columna
            features = np.expand_dims(features.flatten(order='F'),axis=0)
            #se agregan los resultados de dffcc y dffc2 a features
            features= np.append(features,np.concatenate((np.expand_dims(np.mean(dfcc,1),axis=0),np.expand_dims(np.mean(dfcc2,1),axis=0)),axis=1),axis=1)
            features = np.transpose(features)

            if tf>ti and fff>fi:

                lista_aux1=[np.int16(fechas.T[0,2:6])]
                lista_aux1=np.array(lista_aux1)
                lista_aux2=np.concatenate((np.expand_dims(ti,axis=0),np.expand_dims(tf,axis=0),np.expand_dims(tf-ti,axis=0),
                                        np.expand_dims(dom,axis=0),np.expand_dims(fi,axis=0),np.expand_dims(fff,axis=0),
                                        np.expand_dims(band_1,axis=0),np.expand_dims(band_2,axis=0)))
                lista_aux2=np.array(lista_aux2)
                lista_aux3=np.append(lista_aux1,lista_aux2)
                lista_aux4=np.append(lista_aux3,features.T)

                segment_data.append(lista_aux4)
                nombre_archivo.append(fechas[0,contador_archivos])
                
            else:
                0
        except:
            print("exepcion en archivo: ", archivo, "Numero: ",k)
segment_data=np.array(segment_data)
nombre_archivo=np.array(nombre_archivo)
nombre_archivo=np.expand_dims(nombre_archivo,axis=1)