import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler 
from functions import data_structure, detection

import sys
#Ejemplo de path
#sys.path.insert(1, 'E:\Documents\Doctorado\Codigos_CUVI\Descompensaciones_Victoria\Scripts')

def decompesation_analisis(path,path2,agregado, start,end,grupos,edificio):
    '''
    :param path: ruta
    :param start: fecha inicio periodo a analizar
    :param end: fecha final periodo a analizar
    :param grupos: numero de grupos para cluster; si va vacío si hará un análisi para ver el número óptimo
    :param edificio: Derechos o Villabuena
    :return: We define the building data based on its architecture
    '''

    #En base al path, si es agregado y el año de inicio y de final unimos datos
    consumos, t_int,t_ext, radiation = data_structure(path,agregado, start,end)
    if agregado==True:
        year='agregado'
    else:
        year=str(pd.to_datetime(start).year)

    nombres=t_int.columns
    t_out = t_ext.iloc[:,4]
    rad=radiation.iloc[:,4]

    #Activar por si vienen datos vacíos mal formateados
    t_int = t_int.replace(" ", np.nan)
    for t in range(t_int.shape[1]):
        t_int.iloc[:,t]=pd.to_numeric(t_int.iloc[:,t])
        t_int.iloc[:,t].interpolate()

    #Creamos variables externas
    m2 = pd.DataFrame(np.tile(np.array([84,64,84]),42)) #metros cuadrados por piso
    exterior = pd.DataFrame(np.transpose(np.tile(t_out, (126,1)))) #matrix de t_exterior igual para todos los pisos
    exterior.index = t_int.index
    exterior.columns = t_int.columns
    diff= t_int-exterior #calculamos salto térmico
    diff.index=consumos.index
    horas = pd.DataFrame(consumos>0).sum(axis=0) #calculamos las horas de consumos por piso
    ############################################
    #Detecciones de pisos con datos malos o extraños -- solucionamos cogiendo la media de sus entornos!!
    o1 = np.where((diff.isna().sum(axis=0) > diff.shape[0] / 2) | (pd.DataFrame(diff == 0).sum(axis=0) > diff.shape[0] / 2))[0] #muchos NaNs o valores igual a 0 en el salto térmico
    o2 = np.where((consumos.isna().sum(axis=0) > int(consumos.shape[0]*0.75)))[0] #valores NaNs más que el 75% de los datos considerados

    if len(o1) > 0:
        for g in range(len(o1)):
            #Derechos humanos
            if o1[g]<=71:
                if o1[g] in [3, 6, 9, 12,15, 18]:
                    diff.iloc[:,o1[g]] = pd.concat([diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                elif o1[g] in [68, 65, 62, 59, 56, 53]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                elif o1[g] in [22, 46, 70]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] - 1]],axis=1).mean(axis=1)
                elif o1[g] in [1, 25, 49]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3]],axis=1).mean(axis=1)
                elif o1[g] in [2, 26]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] + 22]],axis=1).mean(axis=1)
                elif o1[g] in [45, 69]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] - 22]],axis=1).mean(axis=1)
                elif o1[g] in [24, 48]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 22]],axis=1).mean(axis=1)
                elif o1[g] in [5, 8, 11, 14, 17, 20, 29, 32, 35, 38, 41, 45]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] + 22]],axis=1).mean(axis=1)
                elif o1[g] in [27, 30, 33, 36, 39, 42, 51, 54, 57, 60, 63, 66]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] - 22]],axis=1).mean(axis=1)
                elif o1[g] in [23, 47]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] + 22]],axis=1).mean(axis=1)
                elif o1[g] == 0:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] + 3]],axis=1).mean(axis=1)
                elif o1[g] == 21:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                elif o1[g] == 50:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3]],axis=1).mean(axis=1)
                elif o1[g] == 71:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                else:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] + 1]],axis=1).mean(axis=1)
            else:
                #Villabuena
                oo=o1-72
                if oo[g] in [3, 6, 9, 12,15]:
                    diff.iloc[:,o1[g]] = pd.concat([diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                elif oo[g] in [38, 41, 44, 47]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                elif oo[g] in [16, 34, 52]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] - 1]],axis=1).mean(axis=1)
                elif oo[g] in [1, 19, 37]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3]],axis=1).mean(axis=1)
                elif oo[g] in [2, 20]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] + 16]],axis=1).mean(axis=1)
                elif oo[g] in [33, 51]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] - 16]],axis=1).mean(axis=1)
                elif oo[g] in [18, 36]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g]+- 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 16]],axis=1).mean(axis=1)
                elif oo[g] in [5, 8, 11, 14, 23, 26, 29, 32]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] + 16]],axis=1).mean(axis=1)
                elif oo[g] in [21, 24, 27, 30, 39, 42, 45, 48]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] - 16]],axis=1).mean(axis=1)
                elif oo[g] in [17, 35]:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] + 16]],axis=1).mean(axis=1)
                elif oo[g] == 0:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] + 3]],axis=1).mean(axis=1)
                elif oo[g] == 15:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] + 1], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                elif oo[g] == 38:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3]],axis=1).mean(axis=1)
                elif oo[g] == 53:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] - 3]],axis=1).mean(axis=1)
                else:
                    diff.iloc[:,o1[g]] = pd.concat(
                        [diff.iloc[:,o1[g] - 1], diff.iloc[:,o1[g] + 3], diff.iloc[:,o1[g] - 3], diff.iloc[:,o1[g] + 1]],axis=1).mean(axis=1)

    ############################################
    #CALCULOS
    diff=diff.interpolate(axis=0) #interpolamos por si quedan valores perdidos

    #Creamos matrices vacías
    var = np.zeros((consumos.shape[0], consumos.shape[1]))
    var_con = np.zeros((consumos.shape[0], consumos.shape[1]))

    #Detectamos pisos que tienen menos de 5 horas de consumos para luego no tenerlos en cuenta en la última detección
    o=np.where(horas.reset_index(drop=True)<5)[0]
    o_bool = np.array(horas.reset_index(drop=True)<5)
    

    #Sustituyo los pisos con 0 horas (< 5 ) para no reventar la division
    horas[o]=np.repeat(1,len(np.where(horas.reset_index(drop=True)<5)[0]))

    #Forzamos a tener saltos térmicos vacíos si no hay datos de consumos
    for w in range(consumos.shape[1]):
        diff.iloc[np.where(consumos.iloc[:,w].isna())[0],w]=np.nan

    #Enmascaramientos de valores de salto térmico muy pequeñitos
    diffT = diff.mask(abs(diff) <1, 1)

    #CALCULO KPI (y consumo específico)
    for i in range(var.shape[0]):
        p=np.where(np.array(diffT)[i]<0)[0] #vemos si hat salto termicos negativos
        if len(p)>0:
            var[i,:]=abs((np.array(consumos)[i])/(np.array(m2)[:,0]*np.array(diffT)[i]))
            var_con[i,:]=abs((np.array(consumos)[i])/(np.array(m2)[:,0]))
        else:
            var[i,:]=(np.array(consumos)[i])/(np.array(m2)[:,0]*np.array(diffT)[i])
            var_con[i,:]=(np.array(consumos)[i])/(np.array(m2)[:,0])

    var=pd.DataFrame(var)
    var_con=pd.DataFrame(var_con)


    #Creamos variable que diferencia bloques por pisos (hasta ibamos con los dos bloques)
    bloques = np.concatenate([np.repeat('Derechos',72),np.repeat('Villabuena',54)])
    bloques = pd.DataFrame(bloques).astype('category')

    detection(year,var,var_con,diff,o_bool,t_out,rad,edificio, grupos, bloques,nombres, True, path2)

    print('PISOS SIN DATODS TEMPERATURA:', nombres[o1])
    print('PISOS SIN DATOS DE CONSUMO:', nombres[o2])
    print('PISOS ELIMINADOS POR NO CONSUMOS:', nombres[o])

