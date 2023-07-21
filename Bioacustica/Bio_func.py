import os 
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
import scipy

def fcc5(canto,nfiltros,nc,nframes):    
    """Calculo de FCCs con escalamiento lineal, hace parta de los MFCC 
    (Coeﬁcientes Cepstrales en las Frecuencias de Mel), son coeficientes para la
    representacion del habla basado en la persepcion humana.

    Args:
        canto (_type_): _description_
        nfiltros (_type_): _description_
        nc (_type_): _description_
        nframes (_type_): _description_

    Returns:
        _type_: _description_
    """

    
   # nfiltros = 14
    #nc = 4
    #nframes = 4
    #I11 = sio.loadmat('MCanto.mat')
    #canto = I11['canto']
    a, b = np.shape(canto)
    div = nframes
    w = int(np.floor(b/div))
    b1 = np.empty((a, 0), float)
    for k in range(0, w*div, w):
        bb = np.transpose(np.expand_dims(np.sum(np.power
             (np.abs(canto[:, k:k + w]), 2), axis=1), axis=0))
        b1 = np.append(b1, bb, axis=1)

    if a >= nfiltros:
        _h = np.zeros((nfiltros, a), np.double)
        wf = int(np.floor(a/nfiltros))
        h = np.empty((0, a), float)
        for k in range(0, wf*nfiltros, wf):
            hh = np.expand_dims(fuzz.gaussmf
                 (np.arange(a) + 1, k + wf, wf/4), axis=0)
            h= np.append(h, hh, axis=0)
    fbe = h@b1
    n = nc
    m = nfiltros

    dctm = lambda n, m: np.multiply(np.sqrt(2/m),
           np.cos(np.multiply(np.matlib.repmat
           (np.transpose(np.expand_dims
           (np.arange(n), axis=0)), 1, m),
           np.matlib.repmat(np.expand_dims
           (np.multiply(np.pi, np.arange
           (1, m + 1)-0.5)/m, axis=0), n, 1))))
    dct = dctm(n, m)
    y = dct@np.log(fbe)
    return y

def without_subband_mode_intensities(I1):
    """Adecua la matris de un espectrograma 

    Args:
        I1 (Array): _description_

    Returns:
        _type_: _description_
    """
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

        histii = np.object_(np.histogram(thisi[:], hvec))
        #histii,bins = np.histogram(thisi[:], hvec)
        #histii = np.real(histii)
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

    """
    Determina el valor de la exentricdad de una elipse para saber su morfologica.
    Si es cero es un circulo, si es 1 se aproxima a una linea

    Inputs:
        Elipse: corresponde a la elipse a realizar extraida de opencv
    Returns:
        eccentricity (float): Retorna el valor de la exentricidad de la elipse analizada
    """
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

def seg_xie(intensityi,specgram_time,specgram_frecuency):
    """Realiza en analisis de los elementos de mayor intensidad en el espectrograma para encontrar
    el tiempo y frecuencia maxima y minima de los elementos mas representativos del audio seleccionado

    Args:
        intensity (array): la variable spectrum, salida de la funcion specgram de matplotlib
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
        posicion en el arreglo 2 otorgando el punto inicial y el ancho y alto del elemento. 
        Ejemplo: [posicion_x,posicion_y,ancho,alto]
    """
    specgram_time=np.expand_dims(specgram_time,axis=0)
    specgram_frecuency=np.expand_dims(specgram_frecuency,axis=1)
    specgram_frecuency=np.flipud(specgram_frecuency)
    intensity=intensityi[1:,:]
    spectgram_intensity=20*(np.log10(np.abs(intensity)))#funcion para pasar a desibelios. 
    gauss_intensity = cv2.GaussianBlur(spectgram_intensity,(13,13),sigmaX=2,sigmaY=5)#se utiliza un filtro gausiano. 

    with_suband=without_subband_mode_intensities(gauss_intensity)

    with_suband = with_suband * (with_suband>=0)

    cv2.imwrite("seg_xie.png",with_suband) #guardo la imagen ya que no se puede manipular directamente.

    with_suband=cv2.imread("seg_xie.png",0) #la abro para pocerder con el programa, debo buscar una mejor solucion.
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
        if area>200 and area<40000 and exent>0.3:
            ellipse = cv2.fitEllipse(cnt) #me da los elemetos que componen una elipse
            eccentricity=findeccentricity(ellipse) #uso una funcion que cree para encontrar la exentricidad del elemento

            if eccentricity>0.5:
                spectgram_estructures.append(cnt)  #guardo los elementos que pasan la condicion
        else:
            continue

    segment=[] #Arreglo que pose el tiempo y frecuencia minima y maxima.
    segmentos_nor=[] #pose lo mismo que el anterior pero da la posicion en pixeles
    for element in spectgram_estructures:
        timeI,frecma,duration,magfrec=cv2.boundingRect(element)
        posicion=[int(timeI),int(frecma),int(duration),int(magfrec)]
        segment.append([float(specgram_time[:,(posicion[0]+1)]),float(specgram_time[:,(posicion[0]+posicion[2]-1)]),
        float(specgram_frecuency[(posicion[1]+1),:]),float(specgram_frecuency[(posicion[1]+posicion[3]),:])])
        segmentos_nor.append([posicion[0],posicion[1],posicion[2],posicion[3]])
    segm_xie=np.array(segment)
    segmentos_nor=np.array(segmentos_nor)

    return segm_xie,segmentos_nor

def time_and_date(dir):
    """
    Regresa una lista con las direcciones de cada audio en la carpeta seleccionada asi mismo la fecha
    y hora en la que fue tomado el audio y los devuelve como lista o como diccionario. 

    Args:
        dir (array): Direccion de la carpeta en la que se encuentran los audios a procesar. 

    Returns:
        fechas (array): Devuelve un arreglo en la que cada columna contiene la fecha
        y hora de un audio con el formato requerido.

        cronologia (array): Devuelve lo mismo que fechas pero en un diccionario.

        audios (array): Devuelve la direccion de cada audio encontrado en la carpeta seleccionada.
        
    """
    nombres = os.listdir(dir)
    cronologia={"nombre_archivo":[],
                "Año":[],"Mes":[],"Dia":[],
                "Hora":[],"Minuto":[],"Segundo":[]}

    audios=[]
    fechas=[]

    for name in nombres:
        direccion=dir+"/"+name
        identify_1=name.find("wav")
        identify_2=name.find("mp3")
        if identify_1 !=-1 or identify_2 !=-1:
            audios.append(direccion)
            datos=name.split("_")
            if len(datos) > 2 :
                cronologia["nombre_archivo"].append(name)
                cronologia["Año"].append(datos[1][0:4])
                cronologia["Mes"].append(datos[1][4:6])
                cronologia["Dia"].append(datos[1][6:8])
                cronologia["Hora"].append(datos[2][0:2])
                cronologia["Minuto"].append(datos[2][2:4])
                cronologia["Segundo"].append(datos[2][4:6])
            else:
                cronologia["nombre_archivo"].append(name)
                cronologia["Año"].append("nan")
                cronologia["Mes"].append("nan")
                cronologia["Dia"].append("nan")
                cronologia["Hora"].append("nan")
                cronologia["Minuto"].append("nan")
                cronologia["Segundo"].append("nan")
                
        else:
            0
    fechas.append(cronologia["nombre_archivo"])
    fechas.append(cronologia["Año"])
    fechas.append(cronologia["Mes"])
    fechas.append(cronologia["Dia"])
    fechas.append(cronologia["Hora"])
    fechas.append(cronologia["Minuto"])
    fechas.append(cronologia["Segundo"])
    fechas=np.array(fechas)

    return fechas,cronologia,audios

def ZscoreMV (datos_carac,zscore_min,rel_zscore):
    """Realiza una normalizacion de maximos y minimos a la muestra ingresada

    Args:
        datos_carac (Array): Valores seleccionados de la matriz datos retornada por segmentacion
        zscore_min (Array): el minimo valor en cada columna de datos_carac
        rel_zscore (Array): la diferencia entre el valor maximo (zscore_max) y minimo
        (zscore_min) de cada columna de datos_carac

    Returns:
        out_zscore:(Array): Resultado de la normalizacion
    """
    num_datos,num_feat=datos_carac.shape[:2]
    meansc=np.matlib.repmat(zscore_min,num_datos,1)
    varnsc=np.matlib.repmat(rel_zscore,num_datos,1)
    out_zscore=(datos_carac-meansc)/(varnsc)
    return out_zscore

def segmentacion(ruta,banda,canal):
    
    """Extrae los segmentos significativos en un rango de frecuencia presentes
    en los espectrogramas analizados de la carpeta seleccionada obteniendo 
    sus nombres y frecuencia de muestreo.

    Args:
        ruta (str): Ubicacion de los archivos (carpeta)

        banda (list): Limite inferior y superior de las frecuencias buscadas
        de esta manera banda=[lim_inf,lim_sup]
    
        canal (int): canal en el que se realiza la operación

    Returns:
        segment_data (array): Informacion relevante de todos los segmentos detectados
        desde su posicion en tiempo, duracion, frecuencia, fecha de adquisicion etc.

        nombre_archivo (str array): Nombres de los archivos analizados.
        
        fs (int): frecuencia de muestreo
    """

    fechas,cronologia,audios=time_and_date(ruta)

    #####Funcion segmentacion########
    # esta se programa dentro de primer for con i = 1
    # se asume que el espectrograma genera los datos para s,f,t,p, los cuales son tomados de matlab
    data = sio.loadmat('D:\Proyectos Visual Studio\Python-Jupyther\Bioacustica\Variables\Pspectro.mat')
    s = data['s']
    #y1 = data['y']  
    banda=np.array(banda)
    s = np.abs(s)
    u, v = np.shape(s)
    #resiz=len(y1[:,canal])/len(s[1,:])
    band_1=1/u      # mirar si se usa para fmin
    band_2=1        # mirar si se usa para fmax
    #p=0
    segment_data=[]
    segm_xie_band=np.empty((0,4),float)
    segmentos_nor_band=np.empty((0,4),float)
    contador_archivos=-1
    nombre_archivo=[]

    # Aqui se debe llamar la funcion del espectrograma.

    for archivo in audios:
        contador_archivos=contador_archivos+1

        fs, senial = wavfile.read(archivo)
        frecuency,time,intensity=signal.spectrogram(senial[:,1],fs=fs,nfft=2048,nperseg=569,noverlap=71)
        #intensity,frecuency,time,d=plt.specgram(senial[:,1],Fs=fs,NFFT=2048,noverlap=1550)

        mfband = medfilt2d(s,kernel_size=(5,5))
        selband=np.flip(mfband,axis=0)

        #--------------------------  Xie ----------------------------------------
        
        segm_xie,segmentos_nor=seg_xie(intensity,time,frecuency)

        for k in range(len(segm_xie[:,1])):
            try:
                ti = np.array(segm_xie[k,0]) #tiempo inicial (X)
                tf =np.array(segm_xie[k,1])   #tiempo final(X+W)
                fi = np.array(segm_xie[k,3])    #frecuencia inicial (Y)
                fff = np.array(segm_xie[k,2])    # frecuencia final (Y+H)
                
                if fi>=banda[0] and fff <= banda[1]:
                    segm_xie_band=np.append(segm_xie_band, np.expand_dims(np.array([segm_xie[k,0],segm_xie[k,1],segm_xie[k,2],segm_xie[k,3]]),axis=0), axis=0)
                    segmentos_nor_band=np.append(segmentos_nor_band, np.expand_dims(np.array([segmentos_nor[k,0],segmentos_nor[k,1],segmentos_nor[k,2],segmentos_nor[k,3]]),axis=0), axis=0)     
            except:
                0

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
                0
    segment_data=np.array(segment_data)
    nombre_archivo=np.array(nombre_archivo)
    nombre_archivo=np.expand_dims(nombre_archivo,axis=1)


    return segment_data,nombre_archivo,fs

def mov_std(xmean,nueva_mean,n,std_p):
    a = (nueva_mean-xmean)**2
    b = (std_p**2)/n
    c = (a+b)*(n-1)
    nstd = c**(1/2)
    return nstd

def lamda_unsup(it_num, dat_norma):
   
    """_summary_

    Returns:
        _type_: _description_
    """

    num_datos, num_feat = np.shape(dat_norma)
    num_clases = 1
     #Cluster prototypes initialized with the parameters of the first class (the no-information class (NIC))
    mean_clas = np.ones((1, num_feat))*0.5
    std_clas = np.ones((1,num_feat))*0.25
    conteo=np.array([0])
    
    for t in range(1,it_num+1):
        for j in range(0, num_datos):
           
            mjota=np.matlib.repmat((dat_norma[j,:]),num_clases,1)
            mads = (mean_clas**(mjota))*((1-mean_clas)**(1-mjota))  
            gads = np.prod(mads,axis=1)/(np.prod(mads,axis=1)+(np.prod(1-mads,axis=1)))         
            x,i_gad=(np.max(gads)), np.argmax(gads)
           
            if i_gad==0:
                num_clases = num_clases + 1        
                conteo = np.append(conteo,1)
                mean_clas = np.append(mean_clas,np.expand_dims((np.mean(([mean_clas[0,:],dat_norma[j,:]]),axis=0)),axis=0),axis=0)
                reshh = mov_std(mean_clas[0,:],np.expand_dims((np.mean(([mean_clas[0,:],dat_norma[j,:]]),axis=0)),axis=0),2,std_clas[0,:])
                std_clas = np.append(std_clas,reshh,axis=0)
            else:
                conteo[i_gad]=conteo[i_gad]+1
                new_mean = np.expand_dims(mean_clas[i_gad,:] + (1/(conteo[i_gad]))*(dat_norma[j,:]-mean_clas[i_gad,:]),axis=0)
                std_clas[i_gad,:] = mov_std(mean_clas[i_gad,:],new_mean,conteo[i_gad],std_clas[i_gad,:])
                mean_clas[i_gad,:] = new_mean
        perc = t*100/it_num
    gadso = np.zeros([num_clases,num_datos]) #consider the NIC
    #starts the assignment of data to the classes
    for j in range(0,num_datos):
        mjota=np.matlib.repmat((dat_norma[j,:]),num_clases,1)
        mads = (mean_clas**(mjota))*((1-mean_clas)**(1-mjota))  
        gadso[:,j] = np.prod(mads,axis=1)/(np.prod(mads,axis=1)+(np.prod(1-mads,axis=1)))  
    x, recon = np.expand_dims((np.max(gadso,axis=0)),axis=0), np.expand_dims(np.argmax(gadso,axis=0),axis=0)
    recon[0,0]=0
    a,b=np.shape(recon)
    for k in range(0,b):
        if x[0,k]<0.8:
            # % tener en cuenta que los valores del GAD asignados a la NIC cuando se cumple este umbral, no tienen sentido.
            recon[0,k]=0
    return gadso,recon,mean_clas,std_clas

def compute_icc(dat_norma,mean_clas,gadso):
    #data = sio.loadmat('PComputeICC.mat')
    #training_set = data['trainingSet']
    training_set=dat_norma
    magk=np.size(gadso,0)
    magn=np.size(training_set,0)
    sbe=[]
    mke =np.zeros((np.size(gadso,0), np.size(training_set,1)))
    
    for c in range(0,magk):
      pre_mke = np.zeros((np.shape(training_set)))
      for n in range(0,magn):
        pre_mke[n,:]=numpy.matlib.repmat(gadso[c,n],1, np.size(training_set,1))*training_set[n,:]
      mke[c,:] = np.sum(pre_mke,axis=0)/np.sum(gadso[c,:])
   
    m=np.expand_dims(np.sum(training_set,axis=0)/magn,axis=0)
    pre_sbe=np.empty((0,magn),float)
    sbe=np.empty((0,magk),float)

    for c in range(0,magk):
      for n in range(0,magn):
        pre_sbe=np.append(pre_sbe, gadso[c,n]*np.dot((mke[c,:] -m), np.transpose(mke[c,:] -m)))
      sbe = np.append(sbe,np.sum(pre_sbe))
      pre_sbe=np.empty((0,magn),float)
    sbe = np.sum(sbe) 
    dmin=np.min(scipy.spatial.distance.pdist(mean_clas))
    icc = (sbe/magn)*dmin*np.sqrt(magk)
    return icc

def compute_cv(mean_clas,gadso):
    fila_gadso,columna_gadso=np.shape(gadso)
    pdis_max=np.zeros((0,columna_gadso))
    pdis_min=np.zeros((0,columna_gadso))
    c=0
    z=(c+1)
    
    for c in range(0,fila_gadso-1):
            for z in range(c+1,fila_gadso):
                pdis_max1= np.maximum(np.real(gadso[c,:]),np.real(gadso[z,:]))
                pdis_max= np.append(pdis_max,np.expand_dims(pdis_max1,axis=0),axis = 0)

                pdis_min1=np.minimum(np.real(gadso[c,:]),(gadso[z,:]))
                pdis_min= np.append(pdis_min,np.expand_dims(pdis_min1,axis=0),axis = 0)
   
    dprima=1-np.sum(pdis_min,1)/np.sum(pdis_max,1)
    dmin=np.min(dprima)
    dis=[]
    dk=np.empty((fila_gadso,columna_gadso),float)
    umk = np.max(gadso,1)
    for c in range(0,fila_gadso):
        for n in range(0,columna_gadso):
            dk[c,n]=umk[c]-gadso[c,n]
        dis=np.append(dis,np.expand_dims(1-np.sum(dk[c,:]*np.exp(dk[c,:]))/(columna_gadso*umk[c]*np.exp(umk[c])),axis=0),axis=0)
        
    dis=np.sum(dis)
    cv = (dis/columna_gadso)*dmin*np.sqrt(fila_gadso)
    return cv

def smooth (data):
    matriz=data
    if np.size(matriz) > 2: 
        moving_averages = np.empty((0), float)
        for i in range(0,np.size(matriz)):           
            if i == 0:
                moving_averages = np.append(moving_averages,matriz[i])               
            elif i == 1:
                moving_averages = np.append(moving_averages,(matriz[i-1] + matriz[i] + matriz[i+1])/3)                
            elif (i >= 2) and (i <= (np.size(matriz)-3)):                 
                moving_averages = np.append(moving_averages,(matriz[i-2] + matriz[i-1] + matriz[i] + matriz[i+1] + matriz[i+2])/5)
            elif (i==(np.size(matriz)-2)):
                moving_averages = np.append(moving_averages,(matriz[i-1] + matriz[i] + matriz[i+1])/3)
            else:               
                moving_averages = np.append(moving_averages,(matriz[i]))   
    else:
        moving_averages  = matriz
    return moving_averages

def seleccion_features(it_num,dat_norma):
    
  
    #data = sio.loadmat('Pseleccion_features')
    #ex_level = 1
    #it_num=2
    #dat_norma = data['dat_norma']
    #mad = data['mad']
    #gad= data['gad']
    feat=[]
    feat1=[]
    
    mcv = np.empty((0), float)
    micc = np.empty((0), float)
    gmcv = np.empty((np.size(dat_norma,1),0), float)
    gmicc = np.empty((np.size(dat_norma,1),0), float)
    maxfeat = np.size(dat_norma,1)
    flag=0
    j=0
    k=0
    
    for j in range(0,maxfeat):
        metric3= np.empty((0), float)
        metric4= np.empty((0), float)

        for k in range(0,np.size(dat_norma,1)):
            if np.size(np.nonzero(k==np.array(feat)))==0:
                gadso,recon,mean_clas,std_class= lamda_unsup(it_num, dat_norma[:,feat+[k]])
                mean_clas=np.delete(mean_clas,0,axis=0)
                std_class=np.delete(std_class,0,axis=0)
        
                icc = compute_icc(dat_norma,mean_clas,gadso)
                cv = compute_cv(mean_clas,gadso)                

                metric3 = np.append(metric3,icc)
                metric4 =np.append(metric4,cv)
            else:  
                icc = 0
                cv  = 0                
                metric3 = np.append(metric3,icc)
                metric4 =np.append(metric4,cv)
      
        [dummy, ind] = np.max(metric4),np.argmax(metric4)
        mcv = np.append(mcv,dummy)
        micc = np.append(micc,metric3[ind])
        # se debe crear el vector con la misma dimension del vector que se desea concatenar 
        # y esnecesario emplear expandim y axis=1
        gmcv = np.append(gmcv,np.expand_dims(metric4,axis=1),axis=1)
        gmicc = np.append(gmicc,np.expand_dims(metric3,axis=1),axis=1)
        feat =  feat+[ind]
        #Filtro de promedio movil
        cr = smooth(smooth(mcv))        
        cr = np.diff(cr)

        try:
            
            if np.sum([cr[-11:] < 0]) > 9:
                break 

        except:
            print()      
    [dummy, ind] = np.max(mcv),np.argmax(mcv)
    feat = feat[0:ind+1]
   
    [gadso,recon,mean_clas,std_class] =  lamda_unsup(5, dat_norma[:,feat]) 
    feat=np.array(feat)
    feat=np.expand_dims(np.transpose(feat),axis=0) 
    return feat,gadso,recon,mean_clas,std_class

