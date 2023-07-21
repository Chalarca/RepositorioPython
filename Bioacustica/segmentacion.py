from ctypes import sizeof
#from funciones_aureas import*
from msvcrt import kbhit
from os.path import isfile, join
from scipy.signal import savgol_filter
from scipy.signal import medfilt2d
import numpy as np
import numpy.matlib
import scipy.io as sio
import skfuzzy as fuzz
import fcc5


def segmentacion():
  """Función para segmentar las grabaciones en banda de frecuencia"""
  #[segm_xie,segmentos_nor] = seg_xie([rutain Dir(i).name])

  #####Funcion segmentacion########
  # esta se programa dentro de primer for con i = 1
  # se asume que el espectrograma genera los datos para s,f,t,p, los cuales son tomados de matlab
  

  # Aqui se debe llamar la funcion del espectrograma.
  data = sio.loadmat('Pspectro.mat')
  s = data['s']
  f = data['f']
  t = data['t']
  P = data['P']
  y1 = data['y']

  banda=np.array([2500,3000])
  canal=1
  s = np.abs(s)
  u, v = np.shape(s)
  resiz=len(y1[:,canal])/len(s[1,:])
  band_1=1/u      # mirar si se usa para fmin
  band_2=1        # mirar si se usa para fmax
  p=1; 
  fs=44100

  mfband = medfilt2d(s,kernel_size=(5,5))
  selband=np.flip(mfband,axis=0)

  #--------------------------  Xie ----------------------------------------
  segm_xie_band=np.empty((0,4),float)
  segmentos_nor_band=np.empty((0,4),float)
  #[segm_xie,segmentos_nor] = seg_xie([rutain Dir(i).name]);
  # la funcion seg_xie debe ser revisada y evaluada, ya que ejecuta los datos de la funcion without
  # pero no retorna los datos requeridos desde seg xie

  #segm_xie,segmentos_nor=seg_xie(P,t,f)

  # por lo anterior, se cargan datos de matlab para continuar con la 
  # traduccion del codigo.
  data = sio.loadmat('Pseg_xie2parte.mat')
  segm_xie= data['segm_xie']
  segmentos_nor= data['segmentos_nor']

  for k in range(len(segm_xie[:,1])):
    try:
      ti = np.array(segm_xie[k,0]) #tiempo inicial (X)
      tf =np.array(segm_xie[k,1])   #tiempo final(X+W)
      fi = np.array(segm_xie[k,3])    #frecuencia inicial (Y)
      fff = np.array(segm_xie[k,2])    # frecuencia final (Y+H)
      
      if (fi>=banda[0]) and (fff <= banda[1]):
        segm_xie_band=np.append(segm_xie_band, np.expand_dims(np.array([segm_xie[k,0],segm_xie[k,1],segm_xie[k,2],segm_xie[k,3]]),axis=0), axis=0)
        segmentos_nor_band=np.append(segmentos_nor_band, np.expand_dims(np.array([segmentos_nor[k,0],segmentos_nor[k,1],segmentos_nor[k,2],segmentos_nor[k,3]]),axis=0), axis=0)     
    except:
      print()

  segm_xie = segm_xie_band
  segmentos_nor=segmentos_nor_band

  k=0
  for k in range(len(segm_xie[:,1])): 
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

      features= np.append(((features[:]),(np.mean(dfcc,1)),(np.mean(dfcc2,1))),axis=0)
      
    except:
          print()    
#return           
      
   
 


   



