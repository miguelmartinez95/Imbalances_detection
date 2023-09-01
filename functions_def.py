import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from datetime import datetime


def two_scales(df, ax1, var, var_lab, y_lab1, y_lab2, order):
    ax2 = ax1.twinx()
    g = sns.barplot(data=df, x='Grupos', y=var, edgecolor='black', hue=var_lab, palette=['blue', 'red', 'green'],
                    ax=ax1)
    g.legend_.set_title(None)
    ax1.set_ylabel(y_lab1, fontsize=23)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelsize=21)
    ax1.tick_params(axis='y', labelsize=22)
    if order == 2:
        for i, thisbar in enumerate(g.patches):
            # Set a different hatch for each bar
            thisbar.set_hatch('\\')
        ax1.set_ylim([0, 120])
    else:
        ax1.set_ylim([0, 12])
    ax1.legend(loc='upper left', fontsize=16, fancybox=True, framealpha=0.5)
    g2 = sns.lineplot(data=df, x='Grupos', y='Temp', hue='Temp_lab', marker='o', sort=False, ax=ax2,
                      palette=['blue', 'red', 'green'])
    # g2.legend_.set_title(None)
    ax2.set_ylabel(y_lab2, fontsize=23)
    ax2.set_xlabel('')
    ax2.set_ylim([0, 15])
    ax2.tick_params(axis='x', labelsize=21)
    ax2.tick_params(axis='y', labelsize=22)
    ax2.get_legend().remove()
    # ax2.legend(loc='upper right', fontsize=17,fancybox=True, framealpha=0.5)


def bar_line_plot(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    two_scales(df, ax1, 'KPI', 'kpi_lab', r'KPI (W/m $^{2}$ $\cdot$ $^\circ$C)', r'$\Delta$ T ($^\circ$C)', 1)
    two_scales(df, ax2, 'Cons', 'Cons_lab', r'Consumption (W/m$^{2}$)', r'$\Delta$ T ($^\circ$C)', 2)
    plt.tight_layout(pad=3)
    plt.show()
    # sns.barplot(data=df, x='Grupos', y='KPI', hue='kpi_lab', alpha=0.5, ax=ax1)
    # ax#2 = ax1.twinx()
    # sns.lineplot(data=df['Temp'], hue=df['Temp_lab'], marker='o', sort=False, ax=ax2)


def temporal_plot(dates, var, diff, grupos, lista, imbalances):
    for t in range(grupos):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        kpi, temps = var.iloc[:, lista[t]], diff.iloc[:, lista[t]]
        kpi.index = dates
        ##################################################
        kpi=kpi.resample('3H').sum()
        kpi=kpi.iloc[range(8*15),:]
        temps.index = dates
        temps=temps.resample('3H').mean()
        temps=temps.iloc[range(8*15),:]
        x = [datetime.strptime(str(d), "%Y-%m-%d %H:%M:%S").date() for d in dates]
        ax1.plot(kpi, color='grey')
        ax2.plot(temps, color='grey')

        if len(imbalances[t][0]):
            kpi1, temps1 = kpi.iloc[:, imbalances[t][0]], temps.iloc[:, imbalances[t][0]]
            ax1.plot(kpi.index, kpi1, color='red', linewidth=2, label='Imbalance 1')
            ax2.plot(temps.index, temps1, color='red', linewidth=2)

        if len(imbalances[t][1]):
            kpi2, temps2 = kpi.iloc[:, imbalances[t][1]], temps.iloc[:, imbalances[t][1]]
            ax1.plot(kpi.index, kpi2, color='green', linewidth=2, label='Imbalance 2')
            ax2.plot(temps.index, temps2, color='green', linewidth=2)

        ax1.set_ylabel(r'KPI (W/m $^{2}$ $\cdot$ $^\circ$C)', fontsize=23)
        ax1.set_xlabel('Time', fontsize=23)
        ax1.legend(loc='upper left', fontsize=16, fancybox=True, framealpha=0.5)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax1.tick_params('x', labelrotation=45)
        ax2.set_ylabel(r'$\Delta$ T ($^\circ$C)', fontsize=23)
        ax2.set_xlabel('Time', fontsize=23)
        ax2.legend(loc='upper left', fontsize=16, fancybox=True, framealpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax2.tick_params('x', labelrotation=45)
        plt.show()


def detection(dates, year, var, var_con, diff, o_bool, exterior, rad, grupos, nombres,portales,letras,pisos, save_results,
              path):
    '''
    :param var: KPI
    :param var_con: Consumo específico
    :param diff: Salto termico
    :param o: indices de pisos sin consumos
    :param exterior: Temperatura exterior
    :param rad: Irradiancia
    :param horas:
    :param bloque: Derechos o Villabuena
    :param grupos: Número de clusters
    :param bloques: Factor diferenciando todos los pisos por bloques
    :param nombres:
    :param save_results: Guardamos gráficos de grupos?
    :param path:
    :return: After create the matrix in which each row is a dwelling with its environment k-means is carried out. Base on each group the decompesations detection is performed.
    '''

    print('start')

    ndias = int(var.shape[0] / 24)
    indices = exterior.reset_index(drop=True).index

    # Detección de días con un nivel de radiación por debajo de un limite y fríos
    try:
        rad_split = np.split(rad.loc[:, 'rad'], ndias)
        ext_split = np.split(exterior.loc[:, 'temp'], ndias)
        indices1 = np.split(indices, ndias)
    except:
        rad_split = np.split(rad, ndias)
        ext_split = np.split(exterior, ndias)
        indices1 = np.split(indices, ndias)

    ind_out = []
    for i in range(len(rad_split)):
        r = rad_split[i]
        te = ext_split[i]

        ind = np.where(r > 200)[0]
        ind2 = np.where(te > 15)[0]

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
        dates = np.delete(dates, ind_out)
    ###########################################################################
    # Calculo de valores concretos para cada piso y para cada variable
    horas = pd.DataFrame(var_con > 0).sum(axis=0)
    diff_mean = diff.mean(axis=0, skipna=True)
    var_sum = var.sum(axis=0) / np.array(horas)
    var_con_sum = var_con.sum(axis=0) / np.array(horas)
    var_sum.index = nombres
    var_con_sum.index = nombres
    diff.iloc.index = nombres
    ####################################################################
    # Creamos y rellenamos matriz con consumos y temperatures de cada pisos junto a los saltos termicos de sus vecinos (según bloque)
    df = pd.concat([var_sum, diff_mean], axis=1)
    matrix = np.zeros((df.shape[0], 10))
    matrix[:, 0] = df.iloc[:, 0]
    matrix[:, 1] = df.iloc[:, 1]

    #portales medios
    g, g2 = 0, 0
    ar = list()
    while g < portales:
        ar.append(np.arange(2 + g2, letras + g2, 1))
        g += 1
        g2 += letras
    ar_medios = np.concatenate(ar)
    portalesT = portales*letras
    portal = 1
    piso = 1
    mask_cosumo = -0.01
    mask_temp = 0
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
        if (i + 1) % letras == 0:
            piso += 1
        if (i + 1) in np.arange(letras*pisos, letras*pisos*(4-1)+1, letras*pisos):
            piso = 1
        if (i + 1) % letras == 0 and (i + 1) < letras*pisos:
            portal = 1
        elif (i + 1) % letras == 0 and (i + 1) >= letras*pisos and (i + 1) < letras*pisos*2:
            portal = 4
        elif (i + 1) % letras == 0 and (i + 1) >= letras*pisos*2 and (i + 1) < letras*pisos*3:
            portal = 7
        elif (i + 1) % letras == 0 and (i + 1) >= letras*pisos*3 and (i + 1) < letras*pisos*4:
            portal = 10
        else:
            portal += 1
    # Eliminación de los bajos debido a la falta de datos de sus entornos
    g,g2 = 0,0
    ar = list()
    while g < (portales):
        ar.append(np.arange(0+g2,0+letras+g2))
        g += 1
        g2 += letras * pisos
    ar = np.concatenate(ar)
    matrix = np.delete(matrix, ar, 0)
    var = var.drop(var.columns[ar], axis=1)
    diff = diff.drop(diff.columns[ar], axis=1)
    o_bool = np.delete(o_bool, ar)
    var_con_sum = var_con_sum.drop(var_con_sum.index[ar], axis=0)
    nombres = np.delete(nombres, ar, 0)
    ##################################################################################################
    piso = matrix[:, np.array([0, 1])]
    entorno = matrix[:, np.array([3, 5, 7, 9])]

    df_entorno = pd.DataFrame(entorno)
    df_piso = pd.DataFrame(piso)
    df_piso = df_piso.replace(np.nan, 0)

    if df_entorno.isnull().values.any():
        raise NameError('Missing values in entorno: REVISE DATA')
    if df_piso.isnull().values.any():
        raise NameError('Missing values in piso: REVISE DATA')

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(np.array(df_entorno))

    ####################################################################################
    if not grupos:
        kmeans_kwargs = {
            "init": "random",
            "n_init": 100,
            "max_iter": 500,
            "random_state": 777,
        }

        Sum_of_squared_distances = []
        K = range(1, 20)
        for k in K:
            km = KMeans(n_clusters=k, **kmeans_kwargs)
            km = km.fit(scaled_features)
            Sum_of_squared_distances.append(km.inertia_)

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

    o = np.where(o_bool == True)[0]
    labels_clean = np.delete(kmeans.labels_, o)
    lista = []
    for t in range(grupos):
        # list.append(np.where(kmeans.labels_ == t)[0])
        lista.append(np.where(labels_clean == t)[0])
        a = pd.DataFrame(df_entorno.iloc[np.where(kmeans.labels_ == t)[0], :])
        a1 = a.transpose()
        a1.index = ['Right', 'Up', 'Left', 'Down']
        a1.plot(figsize=(10, 5), kind='bar', color='blue', width=0.2, legend=False, edgecolor='black', fontsize=22,
                rot=0)
        plt.ylim(-3, 25)
        plt.ylabel(r'$\Delta$T [$^\circ$C]', fontsize=22)
        az = pd.DataFrame(df_entorno.iloc[np.where(kmeans.labels_ == t)[0], :])
        a = az[az >= 0].mean(axis=0, skipna=True)
        a.index = ['Right', 'Up', 'Left', 'Down']
        print('GRUPO', t, 'tiene de medias', a)
        a.plot(kind='bar', alpha=0.75, color='red', width=0.2, legend=False, edgecolor='black', figsize=(8, 6),
               fontsize=22, rot=0)
        plt.ylim(-3, 25)

        if save_results == True:
            sep = '\\'
            pp = sep.join([path, year])
            plt.savefig(pp + '\\' + 'g' + str(t) + '.png')

    detection_sup = []
    detection_inf = []
    df_piso.iloc[np.where(df_piso.iloc[:, 0] < 0)[0], 0] = np.repeat(-0.01, len(np.where(df_piso.iloc[:, 0] < 0)[0]))
    kpi_group = [0 for x in range(grupos)]
    kpi_red = [0 for x in range(grupos)]
    kpi_green = [0 for x in range(grupos)]
    Q_group = [0 for x in range(grupos)]
    Q_red = [0 for x in range(grupos)]
    Q_green = [0 for x in range(grupos)]
    t_group = [0 for x in range(grupos)]
    t_red = [0 for x in range(grupos)]
    t_green = [0 for x in range(grupos)]

    df_piso = df_piso.drop(df_piso.index[o], axis=0)
    var_con_sum = var_con_sum.drop(var_con_sum.index[o], axis=0)
    var_clean = var.drop(var.columns[o], axis=1)
    diff_clean = diff.drop(diff.columns[o], axis=1)

    imb1 = []
    imb2 = []
    for z in range(grupos):
        thermal = df_piso.iloc[lista[z], 0].reset_index(drop=True)
        cons_esp = var_con_sum.iloc[lista[z]].reset_index(drop=True)
        temp = df_piso.iloc[lista[z], 1].reset_index(drop=True)
        kpi_group[z] = np.round(np.mean(thermal[thermal > 0]), 6)
        Q_group[z] = np.round(np.mean(cons_esp[cons_esp > 0]), 6)
        t_group[z] = np.round(np.mean(temp[temp > 0]), 6)

        print('Media thermal GRUPO', z, kpi_group[z])
        print('Media consumo_esp GRUPO', z, Q_group[z])
        print('Media temp GRUPO', z, t_group[z])
        names = nombres[lista[z]]

        # Percentiles utilizados para las detecciones
        thermal_mean_O = thermal.iloc[thermal.index[thermal > 0]].quantile(0.75)
        # thermal_mean2_O = thermal.iloc[thermal.index[thermal > 0]].quantile(0.9)
        temp_mean_O = temp.iloc[temp.index[temp > 0]].quantile(0.5)
        # temp_mean2_O = temp.iloc[temp.index[temp > 0]].quantile(0.75)

        # Graficos de barras para analizar los pisos detectados (KPI - Salto termico)

        # Limite para las descompensaciones TIPO 1 y 3
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.barh(np.arange(len(thermal)), thermal * 1000, color='black')

        ax1.set_xlim(0, 10)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)
        ax1.set_yticks([])
        ax1.set_xlabel('KPI [W/(m$^2 \cdot ^\circ$C)]', fontsize=20)
        ax1.set_ylabel('Dwellings', fontsize=20)

        ax2.barh(np.arange(len(temp)), temp, color='black')
        ax2.set_xlim(0, 15)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.set_xlabel(r'$\Delta$T [$^\circ$C]', fontsize=20)
        ax2.set_ylabel('Dwellings', fontsize=20)
        ax2.set_yticks([])

        # Tambien detectamos los pisos muy lejanos a la mediana de pisos con consumos positivos (por arriba) a la vez que no esten en el 25% con mayor salto termico
        d1 = np.where(thermal - thermal.iloc[thermal.index[thermal > 0]].quantile(0.5) > 0.003)[0]
        if len(d1) > 0:
            t1 = np.where(temp[d1] < temp.iloc[temp.index[temp > 0]].quantile(0.75))[0]
            if len(t1 > 0):
                candidates0 = d1[t1]

        candidates = np.where(thermal > thermal_mean_O)[0]
        candidates2 = np.where(temp < temp_mean_O)[0]
        candidates_final1 = np.intersect1d(candidates, candidates2)
        imb1.append(candidates_final1)

        try:
            candidates_final1 = np.union1d(candidates_final1, candidates0)
            del candidates0
        except:
            pass

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 1: KPI, Salto térmico y consumo especifíco
        detection_sup.append(names[candidates_final1])
        if len(candidates_final1) > 0:
            ax1.barh(candidates_final1, thermal.iloc[candidates_final1] * 1000, color='red')
            ax2.barh(candidates_final1, temp.iloc[candidates_final1], color='red')
            kpi_red[z] = np.round(np.mean(thermal.iloc[candidates_final1][thermal.iloc[candidates_final1] > 0]), 6)
            t_red[z] = np.round(np.mean(temp.iloc[candidates_final1][temp.iloc[candidates_final1] > 0]), 6)
            Q_red[z] = np.round(np.mean(cons_esp.iloc[candidates_final1][cons_esp.iloc[candidates_final1] > 0]), 6)

            print('GRUPO', z)
            print('Media KPI de KPI alta y T baja', kpi_red[z]
                  )
            print('Media Salto termico de KPI alta y T baja',
                  Q_red[z])
            print('Media consumo especifico de KPI alta y T baja',
                  t_red[z])

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 2: KPI, Salto térmico y consumo especifíco
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
        thermal_mean = thermal.iloc[thermal.index[thermal > 0]].quantile(0.25)
        # thermal_mean2 = thermal.iloc[thermal.index[thermal > 0]].quantile(0.1)
        temp_mean = temp.iloc[temp.index[temp > 0]].quantile(0.5)
        # temp_mean2 = temp.iloc[temp.index[temp > 0]].quantile(0.25)
        #

        # Tambien detectamos los pisos muy lejanos a la mediana de pisos con consumos positivos (por abajo) a la vez que no esten en el 25% con menor salto termico
        d1 = np.where(thermal - thermal.iloc[thermal.index[thermal > 0]].quantile(0.5) < -0.003)[0]
        if len(d1) > 0:
            t1 = np.where(temp[d1] > temp.iloc[temp.index[temp > 0]].quantile(0.25))[0]
            if len(t1 > 0):
                candidates0 = d1[t1]

        candidates = np.where(thermal < thermal_mean)[0]
        candidates2 = np.where(temp > temp_mean)[0]
        candidates_final2 = np.intersect1d(candidates, candidates2)
        imb2.append(candidates_final2)

        try:
            candidates_final2 = np.union1d(candidates_final2, candidates0)
            del candidates0
        except:
            pass

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 3: KPI, Salto térmico y consumo especifíco
        detection_inf.append(names[candidates_final2])
        if len(candidates_final2) > 0:
            ax1.barh(candidates_final2, thermal.iloc[candidates_final2] * 1000, color='green')
            ax2.barh(candidates_final2, temp.iloc[candidates_final2], color='green')
            kpi_green[z] = np.round(np.mean(thermal.iloc[candidates_final2][thermal.iloc[candidates_final2] > 0]), 6)
            t_green[z] = np.round(np.mean(temp.iloc[candidates_final2][temp.iloc[candidates_final2] > 0]), 6)
            Q_green[z] = np.round(np.mean(cons_esp.iloc[candidates_final2][cons_esp.iloc[candidates_final2] > 0]), 6)

            print('GRUPO', z)
            print('Media KPI de KPI baja y T alta', kpi_green[z]
                  )
            print('Media Salto termico de KPI baja y T alta', Q_green[z]
                  )
            print('Media consumo especifico de KPI baja y T alta', t_green[z]
                  )

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

        # Limite para las descompensaciones TIPO 1 y 3
        ax1.axvline(x=thermal_mean_O * 1000, linewidth=2, color='red')
        # ax1.axvline(x=thermal_mean2_O*1000, linewidth=2, color='red', linestyle='dashed')
        ax2.axvline(x=temp_mean_O, linewidth=2, color='red')
        # ax2.axvline(x=temp_mean2_O, linewidth=2, color='red',linestyle='dashed')

        # Limite para las descompensaciones TIPO 2 y 4
        ax1.axvline(x=thermal_mean * 1000, linewidth=2, color='green')
        # ax1.axvline(x=thermal_mean2*1000, linewidth=2, color='green', linestyle='dashed')
        ax2.axvline(x=temp_mean, linewidth=2, color='green')
        # ax2.axvline(x=temp_mean2, linewidth=2, color='green', linestyle='dashed')

        fig.tight_layout(pad=2.0)
        if save_results == True:
            sep = '\\'
            pp = sep.join([path, year])
            print(pp + '\\' + 'g' + str(z) + 'detec' + '.png')
            plt.savefig(pp + '\\' + 'g' + str(z) + 'detec' + '.png')

    # Printeamos los pisos que forman cada grupos además de los pisos detectados en las posibles descompesaciones
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

    kpi_final = np.concatenate((kpi_group, kpi_red, kpi_green)).reshape(-1, 1)
    temp_final = np.concatenate((t_group, t_red, t_green)).reshape(-1, 1)
    Q_final = np.concatenate((Q_group, Q_red, Q_green)).reshape(-1, 1)
    # l1=np.concatenate((np.repeat('KPI', grupos), np.repeat('KPI i1', grupos), np.repeat('KPI i2', grupos))).reshape(-1,1)
    l1 = np.concatenate((np.repeat('Avg', grupos), np.repeat('i1', grupos), np.repeat('i2', grupos))).reshape(-1, 1)
    l2 = np.concatenate((np.repeat('Avg', grupos), np.repeat('i1', grupos), np.repeat('i2', grupos))).reshape(-1, 1)
    l3 = np.concatenate((np.repeat(r'$\Delta T$', grupos), np.repeat(r'$\Delta T$ i1', grupos),
                         np.repeat(r'$\Delta T$ i2', grupos))).reshape(-1, 1)
    l4 = []
    for t in range(grupos):
        text = 'Group' + str(t + 1)
        l4.append(text)
    l4 = np.array(l4)
    l4 = np.tile(l4, int(len(kpi_final) / grupos)).reshape(-1, 1)
    df_final = pd.DataFrame(np.concatenate((kpi_final * 1000, l1, temp_final, l3, Q_final * 1000, l2, l4), axis=1))
    df_final.columns = ['KPI', 'kpi_lab', 'Temp', 'Temp_lab', 'Cons', 'Cons_lab', 'Grupos']
    df_final.loc[:, ['KPI', 'Temp', 'Cons']] = df_final.loc[:, ['KPI', 'Temp', 'Cons']].apply(pd.to_numeric,
                                                                                              errors='coerce', axis=1)
    df_final.replace(0, np.nan, inplace=True)

    bar_line_plot(df_final)

    temporal_plot(dates, var, diff, grupos, lista, [imb1, imb2])

    print('FINISHED !!')


def data_structure(cp, agregado, start, end):
    sep1 = "\\"
    year = [str(pd.to_datetime(start).year), str(pd.to_datetime(end).year)]

    # Podemos coger todos los años o alguno de ellos (cogiendo dos años para coer los meses de invierno- finales y principios de año)
    if agregado == True:
        cp2 = sep1.join([cp, 'agregado_19-22'])
        consumos = pd.read_csv(sep1.join([cp2, 'consumos.csv']), decimal=',', sep=';', index_col=0)
        t_int = pd.read_csv(sep1.join([cp2, 'temperatures.csv']), decimal=',', sep=';', index_col=0)
        t_out = pd.read_csv(sep1.join([cp2, 't_exterior.csv']), decimal='.', sep=';')
        radiation = pd.read_csv(sep1.join([cp2, 'radiation.csv']), decimal='.', sep=';')
        t_out.index = pd.to_datetime(consumos.index)
        radiation.index = pd.to_datetime(consumos.index)

        dates = pd.to_datetime(consumos.index, format='%d/%m/%Y %H:%M')
        stop = np.where(dates == '2022-02-06 23:00:00')[0][0]
        consumos = consumos.iloc[range(stop + 1)]
        t_ext = t_out.iloc[range(stop + 1)]
        t_int = t_int.iloc[range(stop + 1)]
        radiation = radiation.iloc[range(stop + 1)]
        t_int = t_int.replace(',', '.', regex=True)
    else:
        for t in range(2):
            cp2 = sep1.join([cp, year[t]])
            consumos = pd.read_csv(sep1.join([cp2, 'consumos.csv']), decimal=',', sep=';', index_col=0)
            t_int = pd.read_csv(sep1.join([cp2, 'temperatures.csv']), decimal=',', sep=';', index_col=0)
            t_out = pd.read_csv(sep1.join([cp2, 't_exterior.csv']), decimal=',', sep=';')
            radiation = pd.read_csv(sep1.join([cp2, 'radiation.csv']), decimal=',', sep=';')
            t_out.index = pd.to_datetime(consumos.index)
            radiation.index = pd.to_datetime(consumos.index)

            dates = pd.to_datetime(consumos.index, format='%d/%m/%Y %H:%M')
            if t == 0:
                ind = np.where(dates == pd.to_datetime(start))[0][0]
                consumos1 = consumos.iloc[range(ind, consumos.shape[0]), :]
                t_int1 = t_int.iloc[range(ind, t_int.shape[0]), :]
                t_int1 = t_int1.replace(',', '.', regex=True)
                t_ext1 = t_out.iloc[range(ind, t_out.shape[0]), :]
                radiation1 = radiation.iloc[range(ind, radiation.shape[0]), :]
            else:
                ind = np.where(dates == pd.to_datetime(end))[0][0]
                consumos2 = consumos.iloc[range(ind + 1), :]
                t_int2 = t_int.iloc[range(ind + 1), :]
                t_int2 = t_int2.replace(',', '.', regex=True)
                t_ext2 = t_out.iloc[range(ind + 1), :]
                radiation2 = radiation.iloc[range(ind + 1), :]

        consumos = pd.concat([consumos1, consumos2], axis=0)
        t_int = pd.concat([t_int1, t_int2], axis=0)
        t_ext = pd.concat([t_ext1, t_ext2], axis=0)
        radiation = pd.concat([radiation1, radiation2], axis=0)

    return (consumos, t_int, t_ext, radiation)



