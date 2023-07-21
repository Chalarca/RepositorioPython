import numpy as np

#Arreglo de los datos que ingresan a Zscore desde los outputs de segmentacion.
#datos_carac1=np.array(a[:,7:10])
#datos_carac=np.zeros((datos_carac1.shape[0],27))
#datos_carac2=np.array(a[:,12:])
#datos_carac[:,0:3]=datos_carac1
#datos_carac[:,3:]=datos_carac2

#zscore_min=np.minimum(datos_carac,0)
#zscore_min=np.expand_dims(np.amin(datos_carac,axis=0),axis=0)
#zscore_max=np.expand_dims(np.amax(datos_carac,axis=0),axis=0)
#rel_zscore=zscore_max-zscore_min

def ZscoreMV (datos_carac,zscore_min,rel_zscore):
    """calcula la desviacion estandar de la muestra

    Args:
        datos_carac (Array): Valores seleccionados de la matriz datos retornada por segmentacion
        zscore_min (_type_): el minimo valor en cada columna de datos_carac
        rel_zscore (_type_): la diferencia entre el valor maximo (zscore_max) y minimo
        (zscore_min) de cada columna de datos_carac

    Returns:
        _type_: _description_
    """
    num_datos,num_feat=datos_carac.shape[:2]
    meansc=np.matlib.repmat(zscore_min,num_datos,1)
    varnsc=np.matlib.repmat(rel_zscore,num_datos,1)
    out_zscore=(datos_carac-meansc)/(varnsc)
    return out_zscore


