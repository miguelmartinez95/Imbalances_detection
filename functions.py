import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler 

def detection(year,var,var_con,diff,o_bool,exterior,rad,bloque, grupos, bloques,nombres, save_results, path):
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
    if bloque == 'Derechos':
        var = var.iloc[:,np.where(bloques == 'Derechos')[0]]
        var_con = var_con.iloc[:,np.where(bloques == 'Derechos')[0]]
        nombres = nombres[np.where(bloques == 'Derechos')[0]]
        diff = diff.iloc[:,np.where(bloques == 'Derechos')[0]]
        o_bool = o_bool[np.where(bloques == 'Derechos')[0]]
    else:
        var = var.iloc[:,np.where(bloques == 'Villabuena')[0]]
        var_con = var_con.iloc[:,np.where(bloques == 'Villabuena')[0]]
        nombres = nombres[np.where(bloques == 'Villabuena')[0]]
        diff = diff.iloc[:,np.where(bloques == 'Villabuena')[0]]
        o_bool = o_bool[np.where(bloques == 'Villabuena')[0]]

    ndias=int(var.shape[0] / 24)
    indices = exterior.reset_index(drop=True).index


    # Detección de días con un nivel de radiación por debajo de un limite y fríos
    try:
        rad_split = np.split(rad.loc[:, 'rad'], ndias)
        ext_split = np.split(exterior.loc[:, 'temp'], ndias)
        indices1 = np.split(indices, ndias)
    except:
        rad_split = np.split(rad,ndias)
        ext_split = np.split(exterior,ndias)
        indices1 = np.split(indices, ndias)

    ind_out = []
    for i in range(len(rad_split)):
        r = rad_split[i]
        te = ext_split[i]

        ind = np.where(r > 200)[0]
        ind2 = np.where(te >15)[0]

        indt=np.union1d(ind,ind2)
        if len(indt) > 0:
            ind_out.append(indices1[i])

    if len(np.concatenate(ind_out))>0:
        ind_out = np.concatenate(ind_out)
        var = var.drop(var.index[ind_out], axis=0)
        var = var.reset_index(drop=True)
        var_con = var_con.drop(var_con.index[ind_out], axis=0)
        var_con = var_con.reset_index(drop=True)
        diff = diff.drop(diff.index[ind_out], axis=0)
        diff=diff.reset_index(drop=True)
    ###########################################################################
    #Calculo de valores concretos para cada piso y para cada variable
    horas = pd.DataFrame(var_con>0).sum(axis=0)
    diff_mean = diff.mean(axis=0,skipna=True)
    var_sum = var.sum(axis=0)/np.array(horas)
    var_con_sum = var_con.sum(axis=0)/np.array(horas)
    var_sum.index = nombres
    var_con_sum.index = nombres
    diff.iloc.index = nombres
    ####################################################################
    #Creamos y rellenamos matriz con consumos y temperatures de cada pisos junto a los saltos termicos de sus vecinos (según bloque)
    df = pd.concat([var_sum, diff_mean], axis=1)
    matrix = np.zeros((df.shape[0], 10))
    matrix[:, 0] = df.iloc[:, 0]
    matrix[:, 1] = df.iloc[:, 1]

    if bloque == 'Derechos':
        portalesT = 9
        pisosT = 8
        portal = 1
        piso = 1
        mask_cosumo = -0.01
        mask_temp = 0
        for i in range(portalesT * pisosT):
            print('PORTAL', portal)
            print('PISO', piso)

            if piso == 1 and portal == 1:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 8 and portal == 1:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal == 1 and piso > 1 and piso < 8:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal in [2, 5, 8]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 8 and portal in [2, 5, 8]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal in [3, 6]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 22, 0], df.iloc[i + 22, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 8 and portal in [3, 6]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 22, 0], df.iloc[i + 22, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal in [4, 7]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 22, 0], df.iloc[i - 22, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 8 and portal in [4, 7]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 22, 0], df.iloc[i - 22, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal == 9:
                matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 8 and portal == 9:
                matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal == 9 and piso > 1 and piso < 8:
                matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal in [2, 5, 8] and piso > 1 and piso < 8:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal in [3, 6] and piso > 1 and piso < 8:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 22, 0], df.iloc[i + 22, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal in [4, 7] and piso > 1 and piso < 8:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 22, 0], df.iloc[i - 22, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])

            if (i + 1) % 3 == 0:
                piso += 1
            if (i + 1) in [24, 48]:
                piso = 1
            if (i + 1) % 3 == 0 and (i + 1) < 24:
                portal = 1
            elif (i + 1) % 3 == 0 and (i + 1) >= 24 and (i + 1) < 48:
                portal = 4
            elif (i + 1) % 3 == 0 and (i + 1) >= 48:
                portal = 7
            else:
                portal += 1

        #Eliminación de los bajos debido a la falta de datos de sus entornos
        matrix= np.delete(matrix, np.array([0,1,2,24,25,26,48,49,50]),0)
        o_bool = np.delete(o_bool, np.array([0,1,2,24,25,26,48,49,50]))
        var_con_sum= var_con_sum.drop(var_con_sum.index[np.array([0,1,2,24,25,26,48,49,50])],axis=0)
        nombres= np.delete(nombres, np.array([0,1,2,24,25,26,48,49,50]),0)
    ##################################################################################################
    else:
        portalesT = 9
        pisosT = 6
        portal = 1
        piso = 1

        mask_cosumo = -0.01
        mask_temp = 0
        for i in range(portalesT * pisosT):
            print('PORTAL', portal)
            print('PISO', piso)

            if piso == 1 and portal == 1:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 6 and portal == 1:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([-mask_cosumo, mask_temp])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal == 1 and piso > 1 and piso < 6:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal in [2, 5, 8]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 6 and portal in [2, 5, 8]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal in [3, 6]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 16, 0], df.iloc[i + 16, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 6 and portal in [3, 6]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 16, 0], df.iloc[i + 16, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal in [4, 7]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 16, 0], df.iloc[i - 16, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 6 and portal in [4, 7]:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 16, 0], df.iloc[i - 16, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif piso == 1 and portal == 9:
                matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([0, -5])
            elif piso == 6 and portal == 9:
                matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([4, 5])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal == 9 and piso > 1 and piso < 6:
                matrix[i, np.array([2, 3])] = np.array([mask_cosumo, mask_temp])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal in [2, 5, 8] and piso > 1 and piso < 6:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal in [3, 6] and piso > 1 and piso < 6:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 16, 0], df.iloc[i + 16, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 1, 0], df.iloc[i - 1, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])
            elif portal in [4, 7] and piso > 1 and piso < 6:
                matrix[i, np.array([2, 3])] = np.array([df.iloc[i + 1, 0], df.iloc[i + 1, 1]])
                matrix[i, np.array([4, 5])] = np.array([df.iloc[i + 3, 0], df.iloc[i + 3, 1]])
                matrix[i, np.array([6, 7])] = np.array([df.iloc[i - 16, 0], df.iloc[i - 16, 1]])
                matrix[i, np.array([8, 9])] = np.array([df.iloc[i - 3, 0], df.iloc[i - 3, 1]])

            if (i + 1) % 3 == 0:
                piso += 1
            if (i + 1) in [18, 36]:
                piso = 1

            if (i + 1) % 3 == 0 and (i + 1) < 18:
                portal = 1
            elif (i + 1) % 3 == 0 and (i + 1) >= 18 and (i + 1) < 36:
                portal = 4
            elif (i + 1) % 3 == 0 and (i + 1) >= 36:
                portal = 7
            else:
                portal += 1

        #Eliminación de los bajos debido a la falta de datos de sus entornos
        matrix= np.delete(matrix, np.array([0,1,2,18,19,20,36,37,38]),0)
        o_bool= np.delete(o_bool, np.array([0,1,2,18,19,20,36,37,38]))
        var_con_sum = var_con_sum.drop(var_con_sum.index[np.array([0,1,2,18,19,20,36,37,38])], axis=0)
        nombres= np.delete(nombres, np.array([0,1,2,18,19,20,36,37,38]),0)
    ##########################################################################################

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

    o = np.where(o_bool==True)[0]
    labels_clean=np.delete(kmeans.labels_, o)
    list = []
    for t in range(grupos):
        #list.append(np.where(kmeans.labels_ == t)[0])
        list.append(np.where(labels_clean == t)[0])
        a = pd.DataFrame(df_entorno.iloc[np.where(kmeans.labels_ == t)[0], :])
        a1 = a.transpose()
        a1.index = ['Right', 'Up', 'Left', 'Down']
        a1.plot(figsize=(10, 5),kind='bar', color='blue', width=0.2, legend=False, edgecolor='black', fontsize=22,
                rot=0)
        plt.ylim(-3, 25)
        plt.ylabel(r'$\Delta$T [$^\circ$C]', fontsize=22)
        az = pd.DataFrame(df_entorno.iloc[np.where(kmeans.labels_ == t)[0], :])
        a = az[az>=0].mean(axis=0, skipna=True)
        a.index = ['Right', 'Up', 'Left', 'Down']
        print('GRUPO', t, 'tiene de medias', a)
        a.plot(kind='bar', alpha=0.75, color='red', width=0.2, legend=False, edgecolor='black', figsize=(8, 6),
               fontsize=22, rot=0)
        plt.ylim(-3, 25)

        if save_results == True:
            sep='\\'
            pp=sep.join([path,year])
            plt.savefig(pp +'\\'+ 'g'+str(t) + '.png')

    detection_sup = []
    detection_sup_sup = []
    detection_inf = []
    detection_inf_inf = []
    df_piso.iloc[np.where(df_piso.iloc[:, 0] < 0)[0], 0] = np.repeat(-0.01, len(np.where(df_piso.iloc[:, 0] < 0)[0]))
    kpi_group= [0 for x in range(grupos)]
    kpi_red= [0 for x in range(grupos)]
    kpi_green= [0 for x in range(grupos)]
    Q_group= [0 for x in range(grupos)]
    Q_red= [0 for x in range(grupos)]
    Q_green= [0 for x in range(grupos)]
    t_group= [0 for x in range(grupos)]
    t_red= [0 for x in range(grupos)]
    t_green= [0 for x in range(grupos)]
    
    df_piso=df_piso.drop(df_piso.index[o], axis=0)
    var_con_sum = var_con_sum.drop(var_con_sum.index[o], axis=0)


    for z in range(grupos):
        thermal = df_piso.iloc[list[z], 0].reset_index(drop=True)
        cons_esp = var_con_sum.iloc[list[z]].reset_index(drop=True)
        temp = df_piso.iloc[list[z], 1].reset_index(drop=True)
        kpi_group[z]=np.round(np.mean(thermal[thermal > 0]), 6)
        Q_group[z] = np.round(np.mean(cons_esp[cons_esp > 0]), 6)
        t_group[z] = np.round(np.mean(temp[temp > 0]), 6)

        print('Media thermal GRUPO', z, kpi_group[z])
        print('Media consumo_esp GRUPO', z, Q_group[z])
        print('Media temp GRUPO', z, t_group[z])
        names = nombres[list[z]]

        #Percentiles utilizados para las detecciones
        thermal_mean_O = thermal.iloc[thermal.index[thermal > 0]].quantile(0.75)
        #thermal_mean2_O = thermal.iloc[thermal.index[thermal > 0]].quantile(0.9)
        temp_mean_O = temp.iloc[temp.index[temp > 0]].quantile(0.5)
        #temp_mean2_O = temp.iloc[temp.index[temp > 0]].quantile(0.75)

        #Graficaos de barras para analizar los pisos detectados (KPI - Salto termico)
        #Limite para las descompensaciones TIPO 1 y 3
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))
        ax1.barh(np.arange(len(thermal)), thermal*1000, color='black')

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

        #Tambien detectamos los pisos muy lejanos a la mediana de pisos con consumos positivos (por arriba) a la vez que no esten en el 25% con mayor salto termico
        d1 = np.where(thermal-thermal.iloc[thermal.index[thermal > 0]].quantile(0.5)>0.003)[0]
        if len(d1)>0:
            t1 = np.where(temp[d1]<temp.iloc[temp.index[temp > 0]].quantile(0.75))[0]
            if len(t1>0):
                candidates0 = d1[t1]


        candidates = np.where(thermal > thermal_mean_O)[0]
        candidates2 = np.where(temp < temp_mean_O)[0]
        candidates_final = np.intersect1d(candidates, candidates2)

        try:
            candidates_final = np.union1d(candidates_final, candidates0)
            del candidates0
        except:
            pass

        #Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 1: KPI, Salto térmico y consumo especifíco
        detection_sup.append(names[candidates_final])
        if len(candidates_final) > 0:
            ax1.barh(candidates_final, thermal.iloc[candidates_final]*1000, color='red')
            ax2.barh(candidates_final, temp.iloc[candidates_final], color='red')
            kpi_red[z]= np.round(np.mean(thermal.iloc[candidates_final][thermal.iloc[candidates_final] > 0]), 6)
            Q_red[z]= np.round(np.mean(temp.iloc[candidates_final][temp.iloc[candidates_final] > 0]), 6)
            t_red[z]=np.round(np.mean(cons_esp.iloc[candidates_final][cons_esp.iloc[candidates_final] > 0]), 6)



            print('GRUPO', z)
            print('Media KPI de KPI alta y T baja',kpi_red[z]
                 )
            print('Media Salto termico de KPI alta y T baja',
                  Q_red[z])
            print('Media consumo especifico de KPI alta y T baja',
                  t_red[z])

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 2: KPI, Salto térmico y consumo especifíco
        #candidates = np.where(thermal > thermal_mean2_O)[0]
        #candidates2 = np.where(temp > temp_mean2_O)[0]
        #candidates_final = np.intersect1d(candidates, candidates2)
        #detection_sup_sup.append(names[candidates_final])
        #if len(candidates_final) > 0:
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
        #thermal_mean2 = thermal.iloc[thermal.index[thermal > 0]].quantile(0.1)
        temp_mean = temp.iloc[temp.index[temp > 0]].quantile(0.5)
        #temp_mean2 = temp.iloc[temp.index[temp > 0]].quantile(0.25)
#


        #Tambien detectamos los pisos muy lejanos a la mediana de pisos con consumos positivos (por abajo) a la vez que no esten en el 25% con menor salto termico
        d1 = np.where(thermal-thermal.iloc[thermal.index[thermal > 0]].quantile(0.5)<-0.003)[0]
        if len(d1)>0:
            t1 = np.where(temp[d1]>temp.iloc[temp.index[temp > 0]].quantile(0.25))[0]
            if len(t1>0):
                candidates0 = d1[t1]

        candidates = np.where(thermal < thermal_mean)[0]
        candidates2 = np.where(temp > temp_mean)[0]
        candidates_final = np.intersect1d(candidates, candidates2)

        try:
            candidates_final = np.union1d(candidates_final, candidates0)
            del candidates0
        except:
            pass

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 3: KPI, Salto térmico y consumo especifíco
        detection_inf.append(names[candidates_final])
        if len(candidates_final) > 0:
            ax1.barh(candidates_final, thermal.iloc[candidates_final]*1000, color='green')
            ax2.barh(candidates_final, temp.iloc[candidates_final], color='green')
            kpi_green[z]= np.round(np.mean(thermal.iloc[candidates_final][thermal.iloc[candidates_final] > 0]), 6)
            Q_green[z]= np.round(np.mean(temp.iloc[candidates_final][temp.iloc[candidates_final] > 0]), 6)
            t_green[z]=np.round(np.mean(cons_esp.iloc[candidates_final][cons_esp.iloc[candidates_final] > 0]), 6)



            print('GRUPO', z)
            print('Media KPI de KPI baja y T alta',kpi_green[z]
                  )
            print('Media Salto termico de KPI baja y T alta',Q_green[z]
                  )
            print('Media consumo especifico de KPI baja y T alta',t_green[z]
                  )

        # Printeamos info de cada uno de los grupos en base a la descompesacion TIPO 4: KPI, Salto térmico y consumo especifíco
        #candidates = np.where(thermal < thermal_mean2)[0]
        #candidates2 = np.where(temp < temp_mean2)[0]
        #candidates_final = np.intersect1d(candidates, candidates2)
        #detection_inf_inf.append(names[candidates_final])
        #if len(candidates_final) > 0:
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
        ax1.axvline(x=thermal_mean_O*1000, linewidth=2, color='red')
        #ax1.axvline(x=thermal_mean2_O*1000, linewidth=2, color='red', linestyle='dashed')
        ax2.axvline(x=temp_mean_O, linewidth=2, color='red')
        #ax2.axvline(x=temp_mean2_O, linewidth=2, color='red',linestyle='dashed')

        #Limite para las descompensaciones TIPO 2 y 4
        ax1.axvline(x=thermal_mean*1000, linewidth=2, color='green')
        #ax1.axvline(x=thermal_mean2*1000, linewidth=2, color='green', linestyle='dashed')
        ax2.axvline(x=temp_mean, linewidth=2, color='green')
        #ax2.axvline(x=temp_mean2, linewidth=2, color='green', linestyle='dashed')




        fig.tight_layout(pad=2.0)
        if save_results == True:
            sep='\\'
            pp=sep.join([path,year])
            print(pp +'\\'+ 'g'+str(z)+'detec' + '.png')
            plt.savefig(pp +'\\'+ 'g'+str(z)+'detec' + '.png')


    #Printeamos los pisos que forman cada grupos además de los pisos detectados en las posibles descompesaciones
    for g in range(grupos):
        print('################# GRUPOS ###############')
        print('GRUPO', g)
        print(nombres[list[g]])
        print('################# DETECTIONS ###############')
        print('GRUPO', g)
        print('kWh altos y Tº baja:', detection_sup[g])
        #print('kWh altos y Tº altos:', detection_sup_sup[g])
        print('kWh bajos y Tº altos:', detection_inf[g])
        #print('kWh bajos y Tº bajos:', detection_inf_inf[g])






def data_structure(cp,agregado, start,end):
    sep1="\\"
    year=[str(pd.to_datetime(start).year), str(pd.to_datetime(end).year)]

    #Podemos coger todos los años o alguno de ellos (cogiendo dos años para coer los meses de invierno- finales y principios de año)
    if agregado==True:
        cp2 = sep1.join([cp, 'agregado_19-22'])
        consumos = pd.read_csv(sep1.join([cp2, 'consumos.csv']), decimal=',', sep=';', index_col=0)
        t_int = pd.read_csv(sep1.join([cp2, 'temperatures.csv']), decimal=',', sep=';', index_col=0)
        t_out = pd.read_csv(sep1.join([cp2, 't_exterior.csv']), decimal='.', sep=';')
        radiation = pd.read_csv(sep1.join([cp2, 'radiation.csv']), decimal='.', sep=';')
        t_out.index = pd.to_datetime(consumos.index)
        radiation.index = pd.to_datetime(consumos.index)

        dates = pd.to_datetime(consumos.index, format='%d/%m/%Y %H:%M')
        stop=np.where(dates=='2022-02-06 23:00:00')[0][0]
        consumos=consumos.iloc[range(stop+1)]
        t_ext=t_out.iloc[range(stop+1)]
        t_int=t_int.iloc[range(stop+1)]
        radiation=radiation.iloc[range(stop+1)]
        t_int = t_int.replace(',', '.', regex=True)
    else:
        for t in range(2):
            cp2=sep1.join([cp,year[t]])
            consumos = pd.read_csv(sep1.join([cp2,'consumos.csv']),decimal=',',sep=';', index_col=0)
            t_int = pd.read_csv(sep1.join([cp2,'temperatures.csv']),decimal=',',sep=';',index_col=0)
            t_out = pd.read_csv(sep1.join([cp2,'t_exterior.csv']),decimal=',',sep=';')
            radiation = pd.read_csv(sep1.join([cp2,'radiation.csv']),decimal=',',sep=';')
            t_out.index=pd.to_datetime(consumos.index)
            radiation.index=pd.to_datetime(consumos.index)

            dates = pd.to_datetime(consumos.index,format='%d/%m/%Y %H:%M')
            if t==0:
                ind = np.where(dates == pd.to_datetime(start))[0][0]
                consumos1 = consumos.iloc[range(ind, consumos.shape[0]),:]
                t_int1 = t_int.iloc[range(ind, t_int.shape[0]),:]
                t_int1=t_int1.replace(',','.',regex=True)
                t_ext1 = t_out.iloc[range(ind, t_out.shape[0]),:]
                radiation1 = radiation.iloc[range(ind, radiation.shape[0]),:]
            else:
                ind = np.where(dates == pd.to_datetime(end))[0][0]
                consumos2=consumos.iloc[range(ind+1),:]
                t_int2=t_int.iloc[range(ind+1),:]
                t_int2=t_int2.replace(',','.', regex=True)
                t_ext2=t_out.iloc[range(ind+1),:]
                radiation2=radiation.iloc[range(ind+1),:]

        consumos = pd.concat([consumos1,consumos2], axis=0)
        t_int = pd.concat([t_int1,t_int2], axis=0)
        t_ext = pd.concat([t_ext1,t_ext2], axis=0)
        radiation = pd.concat([radiation1,radiation2], axis=0)

    return(consumos,t_int,t_ext,radiation)



