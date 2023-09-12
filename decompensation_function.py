import pandas as pd
import numpy as np
from functions import  detection

def decompesation_analisis(path2, consumos, t_int, t_out, rad, m2, min_horas, grupos, nombres, letras, portales, pisos, year, dates, smooth, datos_sotano):
    '''
    :param path2: ruta
    :param consumo:datos de consumos de calefacción
    :param t_int: datos de temperaturas interiores
    :param t_out: datos de temperaturas exteriores
    :param rad: datos de irradiancia
    :param m2: vector con los m2 de cada piso
    :param min_horas: horas mínimas de calefacción en el periodo para ser considerado
    :param grupos: numero de grupos para cluster; si va vacío si hará un análisi para ver el número óptimo
    :param nombres: labels para cada todos los pisos
    :param letras: cuantas letras hay por portal
    :param portales: cuantos portales tiene el edificio
    :paramn pisos: cuantos pisos tiene el edifcio
    :param year: año en que comineza el análisis o "agregado"
    :param dates: fechas

    :return: We define the building data based on its architecture
    '''

    # Activar por si vienen datos vacíos mal formateados
    t_int = t_int.replace(" ", np.nan)
    for t in range(t_int.shape[1]):
        t_int.iloc[:, t] = pd.to_numeric(t_int.iloc[:, t])
        t_int.iloc[:, t].interpolate()

    # Creamos variables externas
    exterior = pd.DataFrame(np.transpose(np.tile(t_out, (portales*pisos*letras, 1))))  # matrix de t_exterior igual para todos los pisos
    exterior.index = t_int.index
    exterior.columns = t_int.columns
    diff = t_int - exterior  # calculamos salto térmico
    diff.index = consumos.index
    horas = pd.DataFrame(consumos > 0).sum(axis=0)  # calculamos las horas de consumos por piso
    #################################################
    #Comprobacion consumos entre AC y B
    #################################################

    #test = consumos.sum(axis=0)/horas
    #letras = np.tile(['A','B','C'], int(72/3))
    #bes = test.iloc[np.where(letras=='B')[0]]
    #otros = test.drop(test.index[np.where(letras=='B')[0]], axis=0
    #                  )
    #print('Media AC', np.mean(otros))
    #print('Media B', np.mean(bes))
    #print('Mediana B', np.median(bes))
    #print('Mediana AC', np.median(otros))

    ####################################################

    # Detecciones de pisos con datos malos o extraños -- solucionamos cogiendo la media de sus entornos!!
    o1 = \
    np.where((diff.isna().sum(axis=0) > diff.shape[0] / 2) | (pd.DataFrame(diff == 0).sum(axis=0) > diff.shape[0] / 2))[
        0]  # muchos NaNs o valores igual a 0 en el salto térmico
    o2 = np.where((consumos.isna().sum(axis=0) > int(consumos.shape[0] * 0.75)))[
        0]  # valores NaNs más que el 75% de los datos considerados

    if len(o1) > 0:
        #Calculo del número de pisos que forman ciertos entornos
        # Medio arriba
        g, g2 = 0, 0
        ar = list()
        while g < portales:
            ar.append(np.arange(pisos * letras - 2 - (letras - 3) + g2, pisos * letras - 2 + g2 + 1, 1))
            g += 1
            g2 += letras * pisos
        ar_medio_arriba = np.concatenate(ar).tolist()
        # Medio abajo
        g, g2 = 0, 0
        ar = list()
        while g < portales:
            ar.append(np.arange(letras - 2 - (letras - 3) + g2, letras - 2 + g2 + 1, 1))
            g += 1
            g2 += letras * pisos
        ar_medio_abajo = np.concatenate(ar).tolist()
        # Derecha interior
        g, g2 = 0, 0
        ar = list()
        while g < (portales - 1):
            ar.append(np.arange(letras + letras - 1 + g2, letras * pisos - 1 + g2, letras))
            g += 1
            g2 += letras * pisos
        ar_derecha_interior = np.concatenate(ar).tolist()
        # Izquierda interior
        g, g2 = 0, 0
        ar = list()
        while g < (portales - 1):
            ar.append(np.arange(letras * pisos + letras + g2, letras * pisos * (portales - 1) - letras - 1 + g2, letras))
            g += 1
            g2 += letras * pisos

        ar_izquierda_interior = np.concatenate(ar).tolist()

        for g in range(len(o1)):
                # Izquierda maximo
                if o1[g] in np.arange(letras, pisos * letras - (letras * 2 - 1), letras):
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] + letras], diff.iloc[:, o1[g] - letras]], axis=1).mean(
                        axis=1)
                # Derecha del maximo
                elif o1[g] in np.arange(pisos * letras * (portales - 1) + letras + letras - 1,
                                        pisos * letras * portales - letras, letras):
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] + letras], diff.iloc[:, o1[g] - letras]], axis=1).mean(
                        axis=1)
                # Medio arriba
                elif o1[g] in ar_medio_arriba:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] - letras], diff.iloc[:, o1[g] - 1]], axis=1).mean(
                        axis=1)
                # Medio abajo
                elif o1[g] in ar_medio_abajo:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] + letras]], axis=1).mean(
                        axis=1)
                # Abajo derecha interior
                elif o1[g] in np.arange(letras - 1, (letras - 1 + (pisos * letras) + 1) * (portales - 2),
                                        pisos * letras):
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] + letras], diff.iloc[:, o1[g] + (letras*pisos-2)]], axis=1).mean(
                        axis=1)
                # Arriba izquierda interior
                elif o1[g] in np.arange(pisos * letras * 2 - letras,
                                        (pisos * letras * 2 - letras + (pisos * letras) + 1) * (portales - 2),
                                        pisos * letras):
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] - letras], diff.iloc[:, o1[g] - (letras*pisos-2)]], axis=1).mean(
                        axis=1)
                # Abajo izquierda interior
                elif o1[g] in np.arange(letras * pisos, (letras * pisos + (pisos * letras) + 1) * (portales - 2),
                                        pisos * letras):
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] + letras], diff.iloc[:, o1[g] - (letras*pisos-2)]], axis=1).mean(
                        axis=1)
                # Derecha interior
                elif o1[g] in ar_derecha_interior:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] + letras], diff.iloc[:, o1[g] - letras],
                         diff.iloc[:, o1[g] + (letras*pisos-2)]], axis=1).mean(axis=1)
                # Izquierda interior
                elif o1[g] in ar_izquierda_interior:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] + letras], diff.iloc[:, o1[g] - letras],
                         diff.iloc[:, o1[g] - (letras*pisos-2)]], axis=1).mean(axis=1)
                # Arriba derecha interior
                elif o1[g] in np.arange(letras * pisos - 1,
                                        (letras * pisos - 1 + (pisos * letras * (portales - 2)) + 1), pisos * letras):
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] - letras], diff.iloc[:, o1[g] + (letras*pisos-2)]], axis=1).mean(
                        axis=1)
                # Abajo izquierda maximo
                elif o1[g] == 0:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] + letras]], axis=1).mean(axis=1)
                # Arriba izquierda maximo
                elif o1[g] == pisos * letras - letras:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] + 1], diff.iloc[:, o1[g] - letras]], axis=1).mean(axis=1)
                # Abajo derecha maximo
                elif o1[g] == pisos * letras * (portales - 1) + (letras - 1):
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] + letras]], axis=1).mean(axis=1)
                # Arriba derecha maximo
                elif o1[g] == pisos * letras * portales - 1:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] - letras]], axis=1).mean(axis=1)
                else:
                    diff.iloc[:, o1[g]] = pd.concat(
                        [diff.iloc[:, o1[g] - 1], diff.iloc[:, o1[g] + letras], diff.iloc[:, o1[g] - letras],
                         diff.iloc[:, o1[g] + 1]], axis=1).mean(axis=1)

    ############################################
    # CALCULOS
    diff = diff.interpolate(axis=0)  # interpolamos por si quedan valores perdidos

    # Creamos matrices vacías
    var = np.zeros((consumos.shape[0], consumos.shape[1]))
    var_con = np.zeros((consumos.shape[0], consumos.shape[1]))

    # Detectamos pisos que tienen menos de x horas de consumos para luego no tenerlos en cuenta en la última detección
    o = np.where(horas.reset_index(drop=True) < min_horas)[0]
    o_bool = np.array(horas.reset_index(drop=True) < min_horas)

    # Sustituyo los pisos con 0 horas para no reventar la division
    horas[np.where(horas.reset_index(drop=True) < 1)[0]] = np.repeat(1, len(np.where(horas.reset_index(drop=True) < 1)[0]))

    # Forzamos a tener saltos térmicos vacíos si no hay datos de consumos
    for w in range(consumos.shape[1]):
        diff.iloc[np.where(consumos.iloc[:, w].isna())[0], w] = np.nan

    # Enmascaramientos de valores de salto térmico muy pequeñitos
    diffT = diff.mask(abs(diff) < 2, 1)

    # CALCULO KPI (var) y consumo específico (var_con)
    for i in range(var.shape[0]):
        p = np.where(np.array(diffT)[i] < 0)[0]  # vemos si hat salto termicos negativos
        if len(p) > 0:
            var[i, :] = abs((np.array(consumos)[i]) / (np.array(m2)[:, 0] * np.array(diffT)[i]))
            var_con[i, :] = abs((np.array(consumos)[i]) / (np.array(m2)[:, 0]))
        else:
            var[i, :] = (np.array(consumos)[i]) / (np.array(m2)[:, 0] * np.array(diffT)[i])
            var_con[i, :] = (np.array(consumos)[i]) / (np.array(m2)[:, 0])

    var = pd.DataFrame(var)
    var_con = pd.DataFrame(var_con)

    detection(dates, year, var, var_con, diff, o_bool, t_out, rad, grupos,  nombres, portales, letras, pisos, True, path2, smooth, datos_sotano)

    print('PISOS SIN DATODS TEMPERATURA:', nombres[o1])
    print('PISOS SIN DATOS DE CONSUMO:', nombres[o2])
    print('PISOS ELIMINADOS POR NO CONSUMOS:', nombres[o])

