import os
from ctypes import sizeof
from msvcrt import kbhit
from os.path import isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import scipy.io as sio
import skfuzzy as fuzz
from scipy import signal
from scipy.fft import fftshift
from scipy.io import wavfile
from scipy.signal import medfilt2d, savgol_filter
from Bioacustica_Com1 import ZscoreMV, lamda_unsup, segmentacion, seleccion_features, time_and_date
import pandas as pd


ruta="D:\\Users\ACER\Desktop\Trabajo Investigacion\Aureas Mono especies\Audios"
canal=1
autosel=0
visualize=0
banda=["min","max"]
repre=[]
frecuencia=[]
dispersion=[]

    
if type(banda[0])=="str" and type(banda[1])=="str":
    datos,nombre_archivo,fs=segmentacion(ruta,[0,20000],canal)
else:
    datos,nombre_archivo,fs=segmentacion(ruta,banda,canal)

if visualize==1:
    0
    #funcion que permite la visualizacion de los spectrogramas de cada audio
    #datos,nombre_archivo=VisualizacionSegs(rutain,datos,nombre_archivo,canal,banda)
else:
    0
if len(datos)>0:
    datos_carac1=np.array(datos[:,7:10])
    datos_carac=np.zeros((datos_carac1.shape[0],27))
    datos_carac2=np.array(datos[:,12:])

datos_carac[:,0:3]=datos_carac1
datos_carac[:,3:]=datos_carac2

zscore_min=np.expand_dims(np.amin(datos_carac,axis=0),axis=0)
zscore_max=np.expand_dims(np.amax(datos_carac,axis=0),axis=0)
rel_zscore=zscore_max-zscore_min

datos_clasifi=ZscoreMV (datos_carac,zscore_min,rel_zscore)

infoZC=np.array([zscore_min,zscore_max,0],dtype=object)


if autosel==0:
    feat=np.array(list(range(0,len(datos_clasifi[1]))))
    infoZC[2]=np.expand_dims(feat,axis=0)
    gadso,recon,mean_class,std_class=lamda_unsup(2,datos_clasifi)
    mean_class=mean_class[1:,:] 
    #elimina la primera fila por no ser relevantes
    std_class=std_class[1:,:] #igual

    i=0
    p=0
    ind_eli=[]
    sizeclasses=mean_class.shape[0]
    while p<sizeclasses:
        if sum(recon[0,:]==i)==0:
            ind_eli.append(p)
            recon[recon>i]=recon[recon>i]-1
        else:
            i=i+1
        p=p+1
    mean_class = np.delete(mean_class,ind_eli,0)
    gadso=np.delete(gadso,ind_eli,0)

    for i in range(0,mean_class.shape[0]):
        ind_class=np.where(recon[0,:]==i)[0]
        
        euc=[]
        ind=[]
        for j in ind_class:
            vdat=mean_class[i,:]-datos_clasifi[j,:]
            euc.append(np.dot(vdat,vdat.T))
        [dummy, indm] = np.min(euc),np.argmax(euc)
        #indm siempe (o eso parece) siempre ser 1 tanto en python como en matlab, esto elige un indice
        # que de dejarse asi seria un error en python porque las listas comienzan en 0 y no en uno.
        repre.append(ind_class[indm-1]) 
    mediafrecuencia=[]
    stdfrecuencia=[]


    for i in range(0,mean_class.shape[0]):
        indclass2=np.where(recon[0,:]==i)[0]
        mediafrecuencia.append(np.mean(datos_carac[indclass2],axis=0))
        stdfrecuencia.append(np.std(datos_carac[indclass2],axis=0))

    frecuencia=np.array([mediafrecuencia,stdfrecuencia])
else:
    feat,gadso,recon,mean_class,std_class=seleccion_features(2,datos_clasifi)
    mean_class=mean_class[1:,:] 
    #elimina la primera fila por no ser relevantes
    std_class=std_class[1:,:] #igual
    infoZC[2]=np.expand_dims(feat,axis=0)

    i=0
    p=0
    ind_eli=[]
    sizeclasses=mean_class.shape[0]
    while p<=sizeclasses:
        if sum(recon[0,:]==i)==0:
            ind_eli.append(p)
            recon[recon>i]=recon[recon>i]-1
        else:
            i=i+1
        p=p+1
    mean_class = np.delete(mean_class,ind_eli,0)
    gadso=np.delete(gadso,ind_eli,0)

    for i in range(0,mean_class.shape[0]):
        ind_class=np.where(recon[0,:]==i)[0]
        
        euc=[]
        ind=[]
        for j in ind_class:
            vdat=mean_class[i,:]-datos_clasifi[j,feat]
            euc.append(np.dot(vdat,vdat.T))
        [dummy, indm] = np.min(euc),np.argmax(euc)
        #indm siempe (o eso parece) siempre ser 1 tanto en python como en matlab, esto elige un indice
        # que de dejarse asi seria un error en python porque las listas comienzan en 0 y no en uno.
        repre.append(ind_class[indm-1]) 
    mediafrecuencia=[]
    stdfrecuencia=[]


    for i in range(0,mean_class.shape[0]):
        indclass2=np.where(recon[0,:]==i)[0]
        mediafrecuencia.append(np.mean(datos_carac[indclass2],axis=0))
        stdfrecuencia.append(np.std(datos_carac[indclass2],axis=0))

    frecuencia=np.array([mediafrecuencia,stdfrecuencia])
salida=np.array(np.concatenate([datos[:,0:10],(fs/2)*(datos[:,10:12])],axis=1))
tarr=np.concatenate([salida,np.transpose(recon)],axis=1)
table=np.concatenate([nombre_archivo, tarr],axis=1,dtype="object")

for i in range(0,np.max(recon)):
    dispersion.append(np.sum(np.std(datos_clasifi[(recon[0,:]==i),:],axis=1)))
dispersion=np.expand_dims(np.array(dispersion),axis=0)