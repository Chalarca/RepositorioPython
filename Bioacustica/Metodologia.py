def Metodologia (ruta,banda,canal,autosel,visualize):
    """_summary_

    Args:
        ruta (str): Ruta de la carpeta con los audios a analizar
        banda (list): Limite inferior y superior de las frecuencias buscadas
        de esta manera banda=[lim_inf,lim_sup]
        canal (int): por defecto esta en 1
        autosel (int): elige entre el metodo de lamda o seleccion features
        visualize (int): no implementado aun, por defecto en 0

    Returns:
        table(array): Regresa una talba con todos los datos relevantes analizados
        datos_clasifi(array):Contiene los datos estadisticos realizados por zscore
        mean_class(array):Contiene los datos promedios de los segmentos, analizados
        por la funcion de Lamda
        infoZC(array): Contiene una seleccion especial de datos de los segmentos
        gadso(array): Seleecion de datos provenientes de la funcion lamda
        repre(array): pose los elementos representativos
        dispersion(array):pose la medida de dispersion o desviacion estandar de los
        elementos seleccionados, si habian mas de uno similar se suma.
        frecuencia(array): es un arreglo 3D que contiene el promedio y la desviacion 
        estandar de los datos seleccionados. 
    """

    canal=1
    visualize=0
    repre=[]
    frecuencia=[]
        
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

        i=1
        p=1
        ind_eli=[]
        sizeclasses=mean_class.shape[0]
        while p<=sizeclasses:
            if sum(recon[0,:]==1)==0:
                ind_eli.append(p)
                recon[recon>1]=recon[recon>1]-1
            else:
                i=i+1
            p=p+1
        mean_class = np.delete(mean_class,ind_eli,0)
        gadso=np.delete(gadso,ind_eli,0)

        for i in range(0,mean_class.shape[0]):
            ind_class=np.where(recon[0,:]==i)[0]
            
            euc=[]
            ind=[]
            p=1
            for j in ind_class:
                vdat=mean_class[i,:]-datos_clasifi[j,:]
                euc.append(np.dot(vdat,vdat.T))
                p=p+1
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

        i=1
        p=1
        ind_eli=[]
        sizeclasses=mean_class.shape[0]
        while p<=sizeclasses:
            if sum(recon[0,:]==1)==0:
                ind_eli.append(p)
                recon[recon>1]=recon[recon>1]-1
            else:
                i=i+1
            p=p+1
        mean_class = np.delete(mean_class,ind_eli,0)
        gadso=np.delete(gadso,ind_eli,0)

        for i in range(0,mean_class.shape[0]):
            ind_class=np.where(recon[0,:]==i)[0]
            
            euc=[]
            ind=[]
            p=1
            for j in ind_class:
                vdat=mean_class[i,:]-datos_clasifi[j,feat]
                euc.append(np.dot(vdat,vdat.T))
                p=p+1
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
    return table,datos_clasifi,mean_class,infoZC,gadso,repre,dispersion,frecuencia