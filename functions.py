import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from os import chdir
from pathlib import Path


def to_number(df):
    df = df.replace(' ', np.nan)
    df=df.replace(',','.',regex=True)
    for t in range(df.shape[1]):
        df.iloc[:, t] = pd.to_numeric(df.iloc[:, t])
    return df


def two_scales(df, ax1, var, var_lab, y_lab1, y_lab2, order):
    '''

    :param df: dataframe  con los datos
    :param ax1: eje para realizar un gráfico con matplotlib
    :param var: label for plotted variable (barplot)
    :param var_lab: labels for categories
    :param y_lab1: label for y axisi for plot 1
    :param y_lab2: label for y axisi for plot 1
    :param order: number for the type of the plot (hatch)
    :return: plot of two axis - first for the variables selected and the second for temperatures (pointplot)
    '''

    ax2 = ax1.twinx()
    g = sns.barplot(data=df, x='Grupos', y=var, edgecolor='black', hue=var_lab,
                    palette=['cornflowerblue', 'lightcoral', 'limegreen'],
                    ax=ax1)
    g.legend_.set_title(None)
    ax1.set_ylabel(y_lab1, fontsize=21)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=22)
    ax1.set_ylim([0, np.max(df[var])+int(np.max(df[var])/2)])
    if order == 2:
        for i, thisbar in enumerate(g.patches):
            # Set a different hatch for each bar
            thisbar.set_hatch('\\')
    ax1.legend(loc='upper left', fontsize=16, fancybox=True, framealpha=0.5)

    sns.pointplot(data=df, x='Grupos', y='Temp', hue='Temp_lab', marker='o', sort=False, ax=ax2,
                  palette=['blue', 'red', 'green'], marksize=2, scale=1.2)
    ax2.set_ylabel(y_lab2, fontsize=21)
    ax2.set_xlabel('')
    ax2.set_ylim([0, np.max(df['Temp'])+int(np.max(df['Temp'])/4)])
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=22)
    ax2.get_legend().remove()
    plt.draw()
    plt.pause(0.001)


def bar_line_plot(edificio, df, save_results, path, year):
    '''

    :param df: datos
    :param save_results: guardamos gráficos?
    :param path: localización para guardar gráficos
    :param year: año donde empieza el análisis
    :return: gráficos ajustados y guardados si queremos
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    two_scales(df, ax1, 'KPI', 'kpi_lab', r'KPI (W/m $^{2}$ $\cdot$ $^\circ$C)', r'$\Delta$ T ($^\circ$C)', 1)
    two_scales(df, ax2, 'Cons', 'Cons_lab', r'Consumption (W/m$^{2}$)', r'$\Delta$ T ($^\circ$C)', 2)
    plt.tight_layout(pad=3)

    if save_results == True:
        sep = '\\'
        pp = sep.join([path, 'Graficos'])
        pp = sep.join([pp, year])
        plt.savefig(pp + '\\' + edificio  + 'comparison' + '.png')
        plt.ion()
        plt.show(block=False)


def data_structure(path, agregado, start, end, bloques, bloque):
    '''
    :param cp: path where to find the data
    :param agregado: all the year available?
    :param start: star year (from the indicated month)
    :param end: end yeat (to the indicated month)
    :return: the data joined in a dataframe each variable of interest: consumos, t_interior, t_exterio y radiancia
    '''

    sep1 = "\\"
    path1 = sep1.join([path, 'surfaces.csv'])
    m2 = pd.read_csv(path1, sep=';',
                     decimal='.').loc[:, ['Calle', 'area']]
    m2_complete = np.array(m2.loc[:, 'area'])
    m2 = m2.loc[np.where(m2.iloc[:, 0] == bloque)[0], 'area']

    year = [str(pd.to_datetime(start).year), str(pd.to_datetime(end).year)]
    cp = sep1.join([path, 'Data'])

    # Podemos coger todos los años o alguno de ellos (cogiendo dos años para coger los meses de invierno- finales y principios de año)
    if agregado == True:
        cp2 = sep1.join([cp, 'agregado_19-22'])
        consumos = to_number(pd.read_csv(sep1.join([cp2, 'consumos.csv']), decimal=',', sep=';', index_col=0))
        t_int = to_number(pd.read_csv(sep1.join([cp2, 'temperatures.csv']), decimal=',', sep=';', index_col=0))
        t_out = to_number(pd.read_csv(sep1.join([cp2, 't_exterior.csv']), decimal='.', sep=';'))
        radiation = to_number(pd.read_csv(sep1.join([cp2, 'radiation.csv']), decimal='.', sep=';'))
        t_out.index = pd.to_datetime(consumos.index)
        radiation.index = pd.to_datetime(consumos.index)

        dates = pd.to_datetime(consumos.index, format='%d/%m/%Y %H:%M')
        stop = np.where(dates == '2022-02-06 23:00:00')[0][0]
        consumos = consumos.iloc[range(stop + 1)]
        dates2=dates[range(stop + 1)]
        consumos.index = dates2
        t_ext = t_out.iloc[range(stop + 1)]
        t_ext.index=dates2
        t_int = t_int.iloc[range(stop + 1)]
        t_int.index =dates2
        radiation = radiation.iloc[range(stop + 1)]
        radiation.index=dates2

        place = np.where(bloques == bloque)[0]
        original = pd.DataFrame(consumos)
        t_int_original = pd.DataFrame(t_int)
        consumos = original.iloc[:, place]
        t_int = t_int_original.iloc[:, place]

    else:
        for t in range(2):
            cp2 = sep1.join([cp, year[t]])
            consumos = to_number(
                pd.read_csv(sep1.join([cp2, 'consumos.csv']), decimal=',', sep=';', index_col=0))
            t_int = to_number(pd.read_csv(sep1.join([cp2, 'temperatures.csv']), decimal=',', sep=';', index_col=0))
            t_out = to_number(pd.read_csv(sep1.join([cp2, 't_exterior.csv']), decimal=',', sep=';'))
            radiation = to_number(pd.read_csv(sep1.join([cp2, 'radiation.csv']), decimal=',', sep=';'))

            dates = pd.to_datetime(consumos.index, format='%d/%m/%Y %H:%M')
            if t == 0:
                ind = np.where(dates == pd.to_datetime(start))[0][0]
                consumos1 = consumos.iloc[range(ind, consumos.shape[0]), :]
                t_int1 = t_int.iloc[range(ind, t_int.shape[0]), :]
                t_ext1 = t_out.iloc[range(ind, t_int.shape[0]), :]
                radiation1 = radiation.iloc[range(ind, t_int.shape[0]), :]

                dates2 = dates[range(ind, consumos.shape[0])]
                consumos1.index = dates2
                t_int1.index =  dates2
                t_ext1.index =  dates2
                radiation1.index =  dates2
            else:
                ind = np.where(dates == pd.to_datetime(end))[0][0]
                consumos2 = consumos.iloc[range(ind + 1), :]
                t_int2 = t_int.iloc[range(ind + 1), :]
                t_ext2 = t_out.iloc[range(ind + 1), :]
                radiation2 = radiation.iloc[range(ind + 1), :]

                dates2=dates[range(ind + 1)]
                consumos2.index = dates2
                consumos2.columns = consumos1.columns
                t_int2.index = dates2
                t_int2.columns = t_int1.columns
                t_ext2.index = dates2
                radiation2.index = dates2

        place = np.where(bloques == bloque)[0]
        original = pd.concat([consumos1, consumos2], axis=0)
        t_int_original = pd.concat([t_int1, t_int2], axis=0)
        consumos = pd.concat([consumos1, consumos2], axis=0).iloc[:, place]
        t_int = pd.concat([t_int1, t_int2], axis=0).iloc[:, place]
        t_ext = pd.concat([t_ext1, t_ext2], axis=0)
        radiation = pd.concat([radiation1, radiation2], axis=0)

    return (original, t_int_original, consumos, t_int, t_ext, radiation, m2_complete, m2)


def calculate_dt(t_ext, t_int):
    t_ext = t_ext.loc[:, 'temp']
    exterior = pd.DataFrame(np.transpose(np.tile(t_ext, (t_int.shape[1], 1))))
    exterior.index = t_int.index
    exterior.columns = t_int.columns
    diff = t_int - exterior
    return diff


def check_diff(diff, consumos):
    # Forzamos a tener saltos térmicos vacíos si no hay datos de consumos
    for w in range(diff.shape[1]):
        diff.iloc[np.where(consumos.iloc[:, w].isna())[0], w] = np.nan

    # Enmascaramientos de valores de salto térmico muy pequeñitos
    diffT = diff.mask(abs(diff) < 1, 1)
    return diffT, diff


def calculate_kpi(consumos, diffT, m2):
    # Creamos matrices vacías
    var = np.zeros((consumos.shape[0], consumos.shape[1]))
    var_con = np.zeros((consumos.shape[0], consumos.shape[1]))
    # CALCULO KPI (var) y consumo específico (var_con)
    for i in range(var.shape[0]):
        p = np.where(np.array(diffT)[i] < 0)[0]  # vemos si hat salto termicos negativos
        if len(p) > 0:
            var[i, :] = abs((np.array(consumos)[i]) / (m2 * np.array(diffT)[i]))
            var_con[i, :] = abs((np.array(consumos)[i]) / (m2))
        else:
            var[i, :] = (np.array(consumos)[i]) / (m2 * np.array(diffT)[i])
            var_con[i, :] = (np.array(consumos)[i]) / (m2)

    var = pd.DataFrame(var)
    var_con = pd.DataFrame(var_con)

    return var, var_con


def detec_out_days(dates,var, var_con, diff, radiation, t_ext):
    ndias = int(len(var.index) / 24)
    rad_split = np.split(radiation, ndias)
    te_split = np.split(t_ext, ndias)
    indices1 = np.split(var.index, ndias)

    ind_out = []
    for i in range(len(rad_split)):
        r = rad_split[i]
        te = te_split[i]
        ind = np.where(r > 200)[0]
        ind2 = np.where(te > 14)[0]

        indt = np.union1d(ind, ind2)
        if len(indt) > 0:
            ind_out.append(indices1[i])

    if len(np.concatenate(ind_out)) > 0:
        ind_out = np.concatenate(ind_out)
        var = var.drop(var.index[ind_out], axis=0)
        var = var.reset_index(drop=True)
        var_con = var_con.drop(var_con.index[ind_out], axis=0)
        var_con = var_con.reset_index(drop=True)
        diff = diff.drop(diff.index[ind_out], axis=0)
        diff = diff.reset_index(drop=True)
        dates =np.delete(dates, ind_out)

    days = int(diff.shape[0] / 24)
    print('TOTAL DAYS ANALYSED:', days)
    return var, var_con, diff, dates


def acumulated(diff, var, var_con, nombres, horas):
    ''''
    Pisos con 0 horas de consumos forzamos a 1 para no liar division
    '''
    o = np.where(horas == 0)[0]
    horas.iloc[o] = np.repeat(1, len(np.where(horas == 0)[0]))
    diff = diff.where(diff > 2, np.nan)
    diff_mean = diff.mean(axis=0, skipna=True)
    var_sum = var.sum(axis=0) / np.array(horas)
    var_con_sum = var_con.sum(axis=0) / np.array(horas)
    var_sum.index = nombres
    var_con_sum.index = nombres

    return diff_mean, var_sum, var_con_sum


def create_matrix(var_sum, diff_mean):
    df = pd.concat([var_sum, pd.DataFrame(diff_mean).set_index(var_sum.index)], axis=1)
    matrix = np.zeros((df.shape[0], 10))
    matrix[:, 0] = df.iloc[:, 0]
    matrix[:, 1] = df.iloc[:, 1]

    return matrix, df


def environment_analysis(diff, consumos, pisos, letras, portales,nombres):
    '''
    '''

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
    #print('Mediana AC', np.median(otros))
    #print('Mediana B', np.median(bes))

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

        #Rellenamos los datos de pisos sin datos de temperatura con la media de sus entornos
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

    try:
        print('PISOS SIN DATODS TEMPERATURA:', nombres[o1])
    except:
        raise NameError('All dwelling have data of temperatures')
    try:
        print('PISOS SIN DATOS DE CONSUMO:', nombres[o2])
    except:
        raise NameError('All dwelling have data of consumptions')

    print('FINISHED !!')

    return diff

def d_medios(portales,letras):
    # Números de pisos que se corresponde con portales medios
    g, g2 = 0, 0
    ar = list()
    while g < portales:
        ar.append(np.arange(2 + g2, letras + g2, 1))
        g += 1
        g2 += letras
    ar_medios = np.concatenate(ar)
    return ar_medios

def environment(df, matrix,letras,pisos,portales):
    portal = 1
    piso = 1
    mask_cosumo = -0.01
    mask_temp = 0
    portalesT = portales*letras

    ar_medios = d_medios(portales, letras)

    #Rellenamos matriz con los datos de los entornoos térmicos
    #2-3 consumo y calefacción derecha
    #4-5 consumo y calefacción arriba
    #6-7 consumo y calefacción izquierda
    #8-9 consumo y calefacción abajo
    for i in range(portalesT * pisos):
        print('PORTAL', portal)
        print('PISO', piso)
        if piso == 1 and portal == 1:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([8, 9])] = np.array([0, -5])
        elif piso == pisos and portal == 1:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif portal == 1 and piso > 1 and piso < pisos:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif piso == 1 and portal in ar_medios:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([0, -5])
        elif piso == pisos and portal in ar_medios:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif piso == 1 and portal in np.arange(letras, portalesT-1, letras):
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + (letras*pisos-2), 0], df.iloc[i + (letras*pisos-2), 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([0, -5])
        elif piso == pisos and portal in np.arange(letras, portalesT-1, letras):
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + (letras*pisos-2), 0], df.iloc[i + (letras*pisos-2), 1]])
            matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif piso == 1 and portal in np.arange(letras+1, portalesT-1, letras):
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - (letras*pisos-2), 0], df.iloc[i - (letras*pisos-2), 1]])
            matrix[i, np.array([8, 9])] = np.array([0, -5])
        elif piso == pisos and portal in np.arange(letras+1, portalesT-1, letras):
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - (letras*pisos-2), 0], df.iloc[i - (letras*pisos-2), 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif piso == 1 and portal == portalesT:
            matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([0, -5])
        elif piso == pisos and portal == portalesT:
            matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif portal == portalesT and piso > 1 and piso < pisos:
            matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif portal in ar_medios and piso > 1 and piso < pisos:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif portal in np.arange(letras, portalesT-1, letras) and piso > 1 and piso < pisos:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + (letras*pisos-2), 0], df.iloc[i + (letras*pisos-2), 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])
        elif portal in np.arange(letras+1, portalesT-1, letras) and piso > 1 and piso < pisos:
            matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
            matrix[i, np.array([4, 5])] = np.array([df.iloc[i + letras, 0], df.iloc[i + letras, 1]])
            matrix[i, np.array([6, 7])] = np.array([df.iloc[i - (letras*pisos-2), 0], df.iloc[i - (letras*pisos-2), 1]])
            matrix[i, np.array([8, 9])] = np.array([df.iloc[i - letras, 0], df.iloc[i - letras, 1]])

        #Condiciones para ir reseteando los valores de portal o piso
        if (i + 1) % letras == 0:
            piso += 1
        if (i + 1) in np.arange(letras*pisos, letras*pisos*(portales)+1, letras*pisos):
            piso = 1
        if (i + 1) % letras == 0 and (i + 1) < letras*pisos:
            portal = 1
        elif (i + 1) % letras == 0 and (i + 1) >= letras*pisos and (i + 1) < letras*pisos*2:
            portal = letras+1
        elif (i + 1) % letras == 0 and (i + 1) >= letras*pisos*2 and (i + 1) < letras*pisos*3:
            portal = letras*2+1
        elif (i + 1) % letras == 0 and (i + 1) >= letras*pisos*3 and (i + 1) < letras*pisos*4:
            portal = letras*3+1
        else:
            portal += 1

    # Eliminación de los bajos debido a la falta de datos de sus entornos

    return matrix


def delete_dwellings_no_cons(horas, matrix, var_con_sum, nombres,min_horas):
    o = np.where(horas.reset_index(drop=True) < min_horas)[0]
    if len(o) > 0:
        matrix = np.delete(matrix, o, 0)
        # var_con = var_con.drop(var_con.columns[o], axis=1)
        var_con_sum = var_con_sum.drop(var_con_sum.index[o], axis=0)
        deleted = nombres[o]
        nombres = np.delete(nombres, o, 0)

        print('DWELLINGS DELETED FOR NO CONSUMPTION:', deleted)

    return matrix, var_con_sum, nombres


# Por grupo realizamos un gráfico de barras con los entornos térmicos de cada uno. Juntamos en lista cada uno de los pisos que forman cada grupo.
def plot_environment(df_entorno, grupos, cluster, save_results, path, year, edificio):
    lista = []
    for t in range(grupos):
        lista.append(np.where(cluster.labels_ == t)[0])
        a = pd.DataFrame(df_entorno.iloc[np.where(cluster.labels_ == t)[0], :])
        a1 = a.transpose()
        a1.index = ['Right', 'Up', 'Left', 'Down']
        a1.plot(figsize=(10, 5), kind='bar', color='blue', width=0.2, legend=False, edgecolor='black', fontsize=22,
                rot=0)
        plt.ylim(-3, 18)
        plt.ylabel(r'$\Delta$T [$^\circ$C]', fontsize=22)
        plt.title('Grupo ' + str(t), fontsize=23)
        az = pd.DataFrame(df_entorno.iloc[np.where(cluster.labels_ == t)[0], :])
        a = az[az >= 0].mean(axis=0, skipna=True)
        a.index = ['Right', 'Up', 'Left', 'Down']
        print('GRUPO', t, 'tiene de medias', a)
        a.plot(kind='bar', alpha=0.75, color='red', width=0.2, legend=False, edgecolor='black', figsize=(8, 6),
               fontsize=22, rot=0)
        plt.ylim(-3, 18)
        plt.draw()
        plt.pause(0.001)

        if save_results == True:
            sep = '\\'
            pp = sep.join([path, 'Graficos'])
            pp = sep.join([pp, year])
            plt.savefig(pp + '\\' + edificio + '_g' + str(t) + '.png')
            plt.ion()
            plt.show(block=False)

    return lista


def clustering(df_entorno, grupos):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(np.array(df_entorno))
    ####################################################################################
    # Si no especificamos el nº de grupos buscamos cual podría ser el óptimo y paramos!
    if not grupos:
        kmeans_kwargs = {
            "init": "random",
            "n_init": 100,
            "max_iter": 500,
            "random_state": 6,
        }

        Sum_of_squared_distances = []
        K = range(1, 15)
        for k in K:
            km = KMeans(n_clusters=k, **kmeans_kwargs)
            km = km.fit(scaled_features)
            Sum_of_squared_distances.append(km.inertia_)

        D = abs(np.diff(Sum_of_squared_distances))
        print(D)
        c = []
        z = 'no final'
        for i in range(len(D) - 1):
            if D[i + 1] / D[i] < 0.55:
                z = 'success'
                print('Optimal number of groups:', K[i + 1])
                break
            elif D[i + 1] / D[i] < 0.75:
                c.append(i + 1)

        if z == 'no final' and len(c) > 0:
            print('Condition not fulfill - The optimal number of groups would be: ', c[0])

        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

        raise NameError('Number of groups needed')

    kmeans = KMeans(
        init="random",
        n_clusters=grupos,
        n_init=100,
        max_iter=200,
        random_state=19
    )

    # Agrupamiento con datos escalados
    kmeans.fit(scaled_features)

    return kmeans


def plot_conditions(thermal, temp, z):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_xlim(0, np.max(thermal.values)+np.max(thermal.values)/4)
    ax1.barh(thermal.index, thermal, color='black')

    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_yticks([])
    ax1.set_xlabel('KPI [W/(m$^2 \cdot ^\circ$C)]', fontsize=20)
    ax1.set_ylabel('Dwellings', fontsize=20)

    ax2.set_xlim(0, np.max(temp.values)+np.max(temp.values)/4)
    ax2.barh(temp.index, temp, color='black')
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_xlabel(r'$\Delta$T [$^\circ$C]', fontsize=20)
    ax2.set_ylabel('Dwellings', fontsize=20)
    ax2.set_yticks([])

    fig.suptitle('Grupo ' + str(z), fontsize=22)

    return ax1, ax2, fig


    return ax1, ax2, fig


def drawn_limits(fig, ax1, ax2, thermal_mean_i1, temp_mean_i1, thermal_mean_i2, temp_mean_i2, path, year,save_results, edificio,z):
    # Limit imbalance i1
    ax1.axvline(x=thermal_mean_i1, linewidth=2, color='red')
    ax2.axvline(x=temp_mean_i1, linewidth=2, color='red')
    # Limit imbalance i2
    ax1.axvline(x=thermal_mean_i2, linewidth=2, color='green')
    ax2.axvline(x=temp_mean_i2, linewidth=2, color='green')

    fig.tight_layout(pad=2.0)
    if save_results == True:
        sep = '\\'
        pp = sep.join([path, 'Graficos'])
        pp = sep.join([pp, year])
        plt.savefig(pp + '\\' + edificio + '_g' + str(z) + 'detec' + '.png')


def candidates(thermal, temp, type):
    if type == 'type 1':
        # Percentiles utilizados para las detecciones
        thermal_mean = thermal.iloc[thermal.index[thermal > 0]].quantile(0.75)
        # thermal_mean2_O = thermal.iloc[thermal.index[thermal > 0]].quantile(0.9)
        temp_mean = temp.iloc[temp.index[thermal > 0]].quantile(0.5)
        # temp_mean2_O = temp.iloc[temp.index[temp > 0]].quantile(0.75)

        if len(thermal) >= 3:
            # Tambien detectamos los pisos muy lejanos a la mediana (mediana + 2*std) de pisos con consumos positivos (por arriba) a la vez que no esten en el 25% con mayor salto termico
            d1 = np.where(thermal - (thermal.iloc[thermal.index[thermal > 0]].quantile(0.5) + 2 * (
                np.std(thermal.iloc[thermal.index[thermal > 0]]))) > 0)[0]
            if len(d1) > 0:
                t1 = np.where(temp[d1] < temp.iloc[temp.index[thermal > 0]].quantile(0.75))[0]
                if len(t1 > 0):
                    candidates0 = d1[t1]

            candidates = np.where(thermal > thermal_mean)[0]
            candidates2 = np.where(temp < temp_mean)[0]
            candidates_final = np.intersect1d(candidates, candidates2)
        else:
            candidates_final = []
    else:
        thermal_mean = thermal.iloc[thermal.index[thermal > 0]].quantile(0.25)
        # thermal_mean2 = thermal.iloc[thermal.index[thermal > 0]].quantile(0.1)
        temp_mean = temp.iloc[temp.index[thermal > 0]].quantile(0.5)

        if len(thermal) >= 3:
            # Tambien detectamos los pisos muy lejanos a la mediana de pisos con consumos positivos (por abajo) a la vez que no esten en el 25% con menor salto termico
            d1 = np.where(thermal - (thermal.iloc[thermal.index[thermal > 0]].quantile(0.5) - 2 * (
                np.std(thermal.iloc[thermal.index[thermal > 0]]))) < 0)[0]
            if len(d1) > 0:
                t1 = np.where(temp[d1] > temp.iloc[temp.index[thermal > 0]].quantile(0.25))[0]
                if len(t1 > 0):
                    candidates0 = d1[t1]

            candidates = np.where(thermal < thermal_mean)[0]
            candidates2 = np.where(temp > temp_mean)[0]
            candidates_final = np.intersect1d(candidates, candidates2)
        else:
            candidates_final = []

    try:
        candidates_final = np.union1d(candidates_final, candidates0)
    except:
        candidates_final = candidates_final

    return candidates_final, thermal_mean, temp_mean


def detections(ax1, ax2, thermal, temp, cons_esp, candidates, kpi, temp_list, Q, z, type, detection, names):
    detection.append(names[candidates])
    print('###### GRUPO ######', z)
    if len(candidates) > 0:
        if type=='type 1':
            ax1.barh(thermal.index[candidates], thermal.iloc[candidates], color='red')
            ax2.barh(temp.index[candidates], temp.iloc[candidates], color='red')
        else:
            ax1.barh(thermal.index[candidates], thermal.iloc[candidates], color='green')
            ax2.barh(temp.index[candidates], temp.iloc[candidates], color='green')
        kpi[z] = np.round(np.mean(thermal.iloc[candidates][thermal.iloc[candidates] > 0]), 6)
        temp_list[z] = np.round(np.mean(temp.iloc[candidates][temp.iloc[candidates] > 0]), 6)
        Q[z] = np.round(np.mean(cons_esp.iloc[candidates][cons_esp.iloc[candidates] > 0]), 6)

        print('Media KPI imblance ', type, ': ', kpi[z]
              )
        print('Media Salto termico imbalance ', type, ': ',
              Q[z])
        print('Media consumo especifico imbalance ', type, ': ',
              temp_list[z])
    else:
        print('No imbalances', type, ' detected in GROUP:', z)

    return kpi, Q, temp_list, detection


def groups_info(df_piso, var_con_sum, lista, kpi, temp_list, Q, nombres, z):
    thermal = df_piso.iloc[lista[z], 0].reset_index(drop=True)
    cons_esp = var_con_sum.iloc[lista[z]].reset_index(drop=True)
    temp = df_piso.iloc[lista[z], 1].reset_index(drop=True)
    kpi[z] = np.round(np.mean(thermal[thermal > 0]), 6)
    Q[z] = np.round(np.mean(cons_esp[cons_esp > 0]), 6)
    temp_list[z] = np.round(np.mean(temp[temp > 0]), 6)
    print('##########################################')
    print('Media thermal GRUPO', z, kpi[z])
    print('Media consumo_esp GRUPO', z, Q[z])
    print('Media temp GRUPO', z, temp_list[z])
    names = nombres[lista[z]]
    return thermal, temp, cons_esp, kpi, Q, temp_list, names


def info_detections(grupos, lista,nombres, detection_sup, detection_inf):
    for g in range(grupos):
        print('################# GRUPOS ###############')
        print('GRUPO', g)
        print(nombres[lista[g]])
        print('################# DETECTIONS ###############')
        print('GRUPO', g)
        print('kWh altos y Tº baja:', detection_sup[g])
        # print('kWh altos y Tº altos:', detection_sup_sup[g])
        print('kWh bajos y Tº altos:', detection_inf[g])
        # print('kWh bajos y Tº bajos:', detection_inf_inf[g])


def create_dataframe(kpi, temp, Q, grupos):
    l1 = np.concatenate((np.repeat('Avg', grupos), np.repeat('i1', grupos), np.repeat('i2', grupos))).reshape(-1, 1)
    l2 = np.concatenate((np.repeat('Avg', grupos), np.repeat('i1', grupos), np.repeat('i2', grupos))).reshape(-1, 1)
    l3 = np.concatenate((np.repeat(r'$\Delta T$', grupos), np.repeat(r'$\Delta T$ i1', grupos),
                         np.repeat(r'$\Delta T$ i2', grupos))).reshape(-1, 1)
    l4 = []
    for t in range(grupos):
        text = 'G' + str(t + 1)
        l4.append(text)
    l4 = np.array(l4)
    l4 = np.tile(l4, int(len(kpi) / grupos)).reshape(-1, 1)
    df_final = pd.DataFrame(np.concatenate((kpi, l1, temp, l3, Q, l2, l4), axis=1))
    df_final.columns = ['KPI', 'kpi_lab', 'Temp', 'Temp_lab', 'Cons', 'Cons_lab', 'Grupos']
    df_final.loc[:, ['KPI', 'Temp', 'Cons']] = df_final.loc[:, ['KPI', 'Temp', 'Cons']].apply(pd.to_numeric,
                                                                                              errors='coerce', axis=1)
    df_final.replace(0, np.nan, inplace=True)
    return df_final

def temporal_plot(edificio, dates, var, diff, grupos, lista, imbalances,save_results,path,year, smooth):
    import matplotlib.dates as mdates

    '''

    :param dates: serie para las fechas
    :param var: variable para gráficar
    :param diff: salto térmico de los pisos con el exterior
    :param grupos: número de grupos a buscar por k-means ([] calculamos el número óptimo)
    :param lista: lista con los números de los pisos que forma cada grupo
    :param imbalances: lista con los pisos detectados para las imbalances [0] imbalance type 1 [1] imbalance type 2
    :param save_results: queremos guardar los gráficos?
    :param path: donde guardarlos
    :param year: año donde se empieza el análisis
    :return: gráfico temporal analizando los consumos de los pisos detectados en comparación con los grupos.
    '''
    for t in range(grupos):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        kpi, temps = var.iloc[:, lista[t]], diff.iloc[:, lista[t]]
        kpi.index = dates
        ##################################################
        #Force NaN in dates that are not considered
        st = dates[0]
        end = dates[len(dates)-1]
        new = pd.date_range(st, end, freq='H')
        kpi_new = kpi.reindex(new)
        kpi_new = kpi_new.replace(np.nan, 99999)
        ##################################################
        #Resample for smothing the data in the plot
        if smooth[0]==True:
            kpi_new=kpi_new.resample(smooth[1]).sum() #suma sería
            kpi_new = kpi_new.where(kpi_new <100, np.nan)
            temps.index = dates
            temps=temps.resample(smooth[1]).mean()
        else:
            kpi_new = kpi_new.where(kpi_new < 100, np.nan)

        ax1.plot(kpi_new, color='grey',label='Normal')
        ax2.plot(temps, color='grey',label='Normal')

        #Destacamos detecciones si las hay
        if len(imbalances[0][t]):
            kpi1, temps1 = kpi_new.iloc[:, imbalances[0][t]], temps.iloc[:, imbalances[0][t]]
            ax1.plot(kpi_new.index, kpi1, color='red', linewidth=2, label='Imbalance 1')
            ax2.plot(temps.index, temps1, color='red', linewidth=2,label='Imbalance 1')

        if len(imbalances[1][t]):
            kpi2, temps2 = kpi_new.iloc[:, imbalances[1][t]], temps.iloc[:, imbalances[1][t]]
            ax1.plot(kpi_new.index, kpi2, color='green', linewidth=2, label='Imbalance 2')
            ax2.plot(temps.index, temps2, color='green', linewidth=2, label='Imbalance 2')

        ax1.set_ylabel(r'(W/(h $\cdot$ m $^{2}$)', fontsize=23)
        ax1.set_xlabel('Time', fontsize=20, labelpad=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax1.tick_params('x', labelsize=15, labelrotation=45)
        ax1.tick_params('y', labelsize=15)
        max=kpi.dropna().to_numpy().max()
        ax1.set_ylim([-1, max+max/2.5])

        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(),fontsize=12,loc='upper right')


        ax2.set_ylabel(r'$\Delta$ T ($^\circ$C)', fontsize=23)
        ax2.set_xlabel('Time', fontsize=20, labelpad=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax2.tick_params('x', labelsize=15, labelrotation=45)
        ax2.tick_params('y', labelsize=15)
        max=temps.dropna().to_numpy().max()
        ax2.set_ylim([-1, max+max/2.5])
        fig.suptitle('Grupo '+str(t), fontsize=22)
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), fontsize=12, loc='upper right')
        plt.tight_layout(pad=4)
        plt.draw()
        plt.pause(0.001)

        if save_results == True:
            sep = '\\'
            pp  =sep.join([path, 'Graficos'])
            pp = sep.join([pp, year])
            plt.savefig(pp + '\\' + edificio + '_g' + str(t) + 'temporal' + '.png')
            plt.ion()
            plt.show(block=False)


def deletion(matrix, var_con_sum, nombres, horas, out, datos_sotano,portales,letras,pisos):
    print('Dwellings with too much empty cells: ', nombres[out])

    if datos_sotano == False:
        g, g2 = 0, 0
        ar = list()
        while g < (portales):
            ar.append(np.arange(0 + g2, 0 + letras + g2))
            g += 1
            g2 += letras * pisos
        ar = np.concatenate(ar)
        out_total = np.union1d(out,ar )
    else:
        out_total=out

    if len(out_total) > 0:
        matrix = np.delete(matrix, out_total, axis=0)
        var_con_sum = var_con_sum.drop(var_con_sum.index[out_total])
        horas = horas.drop(horas.index[out_total], axis=0)
        nombres = np.delete(nombres, out_total)


    return matrix, var_con_sum, nombres, horas


def detection_imbalances(df_piso, var_con_sum, lista,nombres,path,year,bloque,save_results):
    detection_sup = []
    detection_inf = []
    df_piso.iloc[np.where(df_piso.iloc[:, 0] < 0)[0], 0] = np.repeat(-0.01, len(np.where(df_piso.iloc[:, 0] < 0)[0]))
    kpi_group = [0 for x in range(len(lista))]
    kpi_red = [0 for x in range(len(lista))]
    kpi_green = [0 for x in range(len(lista))]
    Q_group = [0 for x in range(len(lista))]
    Q_red = [0 for x in range(len(lista))]
    Q_green = [0 for x in range(len(lista))]
    t_group = [0 for x in range(len(lista))]
    t_red = [0 for x in range(len(lista))]
    t_green = [0 for x in range(len(lista))]

    imb1 = []
    imb2 = []

    # Grupo a grupo detectamos los pisos de cada grupo y los límites para detectar imbalances. Se realiza un gráfico de barras mostrando los consumos medios y salto térmico medio
    # comparados con los límites utilizados para las detecciones. Además se caracteriza cada grupo y cada grupo de pisos detectados con valores medios.
    for z in range(len(lista)):
        thermal, temp, cons_esp, kpi_group, Q_group, t_group, names = groups_info(df_piso, var_con_sum, lista,
                                                                                  kpi_group, t_group, Q_group,
                                                                                  nombres, z)

        # Percentiles utilizados para las detecciones
        candidates_final_i1, thermal_mean_i1, temp_mean_i1 = candidates(thermal, temp, 'type 1')
        imb1.append(candidates_final_i1)

        # Graficos de barras para analizar los pisos detectados (KPI - Salto termico)
        ax1, ax2, fig = plot_conditions(thermal, temp, z)

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 1: KPI, Salto térmico y consumo especifíco
        kpi_red, Q_red, t_red, detection_sup = detections(ax1, ax2, thermal, temp, cons_esp, candidates_final_i1,
                                                          kpi_red, t_red,
                                                          Q_red, z, 'type 1', detection_sup, names)

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 3: KPI, Salto térmico y consumo especifíco
        # candidates = np.where(thermal > thermal_mean2_O)[0]
        # candidates2 = np.where(temp > temp_mean2_O)[0]
        # candidates_final = np.intersect1d(candidates, candidates2)
        # detection_sup_sup.append(names[candidates_final])
        # if len(candidates_final) > 0:
        #    ax1.barh(candidates_final, thermal.iloc[candidates_final]*1000, color='blue')
        #    ax2.barh(candidates_final, temp.iloc[candidates_final], color='blue')
        #    print('GRUPO', z)
        #    print('Media KPI de KPI alta y T alta',
        #          np.round(np.mean(thermal.iloc[candidates_final][thermal.iloc[candidates_final] > 0]), 6))
        #    print('Media Salto termico de KPI alta y T alta',
        #          np.round(np.mean(temp.iloc[candidates_final][temp.iloc[candidates_final] > 0]), 6))
        #    print('Media consumo especifico de KPI alta y T alta',
        #          np.round(np.mean(cons_esp.iloc[candidates_final][cons_esp.iloc[candidates_final] > 0]), 6))

        # Otros Percentiles utilizados para las detecciones
        candidates_final_i2, thermal_mean_i2, temp_mean_i2 = candidates(thermal, temp, 'type 2')
        imb2.append(candidates_final_i2)

        kpi_green, Q_green, t_green, detection_inf = detections(ax1, ax2, thermal, temp, cons_esp, candidates_final_i2,
                                                                kpi_green,
                                                                t_green,
                                                                Q_green, z, 'type 2', detection_inf, names)

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 2: KPI, Salto térmico y consumo especifíco
        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 4: KPI, Salto térmico y consumo especifíco
        # candidates = np.where(thermal < thermal_mean2)[0]
        # candidates2 = np.where(temp < temp_mean2)[0]
        # candidates_final = np.intersect1d(candidates, candidates2)
        # detection_inf_inf.append(names[candidates_final])
        # if len(candidates_final) > 0:
        #    ax1.barh(candidates_final, thermal.iloc[candidates_final]*1000, color='purple')
        #    ax2.barh(candidates_final, temp.iloc[candidates_final], color='purple')
        #    print('GRUPO', z)
        #    print('Media KPI de KPI baja y T baja',
        #          np.round(np.mean(thermal.iloc[candidates_final][thermal.iloc[candidates_final] > 0]), 6))
        #    print('Media Salto termico de KPI baja y T baja',
        #          np.round(np.mean(temp.iloc[candidates_final][temp.iloc[candidates_final] > 0]), 6))
        #    print('Media consumo especifico de KPI baja y T baja',
        #          np.round(np.mean(cons_esp.iloc[candidates_final][cons_esp.iloc[candidates_final] > 0]), 6))

        # Limite para las descompensaciones TIPO 1 y 2
        drawn_limits(fig, ax1, ax2, thermal_mean_i1, temp_mean_i1, thermal_mean_i2, temp_mean_i2, path, year,save_results,bloque,z)
        imbalances_ind=  [imb1, imb2]

    return kpi_group, kpi_red, kpi_green, t_group, t_red, t_green, Q_group, Q_red, Q_green, detection_sup, detection_inf, imbalances_ind


def check_data(data, type):
    d=data.copy()
    if type == 'consumption':
        d[d <= 0] = 0
    else:
        d[data <= 1] = np.nan

    o = np.where(d.isna().sum(axis=0) > int(d.shape[0] / 2))[0]

    return data, o


def fix_dataframe(matrix, df, rango):
    o = np.where(np.isnan(matrix[:, 1]))[0]
    df_in = df.copy().iloc[range(rango[0], rango[1] + 1), :]
    if len(o) > 0:
        df_in.iloc[o, 1] = np.nanmean([matrix[o, np.array([3, 5, 7, 9])]])
        df.iloc[range(rango[0], rango[1] + 1), :] = df_in

    return df.copy()


def delete_missing_env(matrix, nombres):
    env = pd.DataFrame(matrix[:, np.array([3, 5, 7, 9])])
    o = np.where(env.isna().sum(axis=1) > 0)[0]
    if len(o):
        matrix = np.delete(matrix, o, axis=0)
        print('Dweeling with missing data in the environment: ', nombres[o])
    else:
        matrix = matrix
        print('All environments correct')

    return (matrix)


def test_data(df_entorno, df_piso):
    if df_entorno.isnull().values.any():
        raise NameError('Missing values in entorno: REVISE DATA')
    if df_piso.isnull().values.any():
        raise NameError('Missing values in piso: REVISE DATA')
