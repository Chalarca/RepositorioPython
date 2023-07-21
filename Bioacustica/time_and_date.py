import os 
import numpy as np

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
    cronologia={"Grabacion #":[],"Formato Grabacion":[],
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
                cronologia["Formato Grabacion"].append(datos[0])
                cronologia["Año"].append(datos[1][0:4])
                cronologia["Mes"].append(datos[1][4:6])
                cronologia["Dia"].append(datos[1][6:8])
                cronologia["Hora"].append(datos[2][0:2])
                cronologia["Minuto"].append(datos[2][2:4])
                cronologia["Segundo"].append(datos[2][4:6])
            else:
                cronologia["Formato Grabacion"].append(name)
                cronologia["Año"].append("nan")
                cronologia["Mes"].append("nan")
                cronologia["Dia"].append("nan")
                cronologia["Hora"].append("nan")
                cronologia["Minuto"].append("nan")
                cronologia["Segundo"].append("nan")
                
        else:
            0
    fechas.append(cronologia["Formato Grabacion"])
    fechas.append(cronologia["Año"])
    fechas.append(cronologia["Mes"])
    fechas.append(cronologia["Dia"])
    fechas.append(cronologia["Hora"])
    fechas.append(cronologia["Minuto"])
    fechas.append(cronologia["Segundo"])
    fechas=np.array(fechas)

    return fechas,cronologia,audios


fechas,cronologia,audios=time_and_date(r'D:\Users\ACER\Desktop\Trabajo Investigacion\Aureas Mono especies\Audios')

print(fechas.T)



