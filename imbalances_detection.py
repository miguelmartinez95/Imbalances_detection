import sys
import urllib.request
import numpy as np
import pandas as pd

#Specific case for GitHub repository
url2 = 'https://raw.githubusercontent.com/miguelmartinez95/Imbalances_detection/main/functions.py'

filename, headers = urllib.request.urlretrieve(url2, filename=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\Scripts\functions.py')

#Ejemplo de path (where the scripts are)
sys.path.insert(1, 'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection')
from functions import data_structure, calculate_dt, check_data, check_diff, calculate_kpi
from functions import detec_out_days, acumulated, create_matrix, environment,environment_analysis
from functions import deletion, delete_dwellings_no_cons, delete_missing_env
from functions import test_data, clustering, plot_environment, detection_imbalances, info_detections, \
    create_dataframe, bar_line_plot

#Localización de datos
path=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection'

#Guardar gráficos
path2 = r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\Graficos'

#Años 2019 y 2020 con horas en 00:07:00, el resto con 00:00:00
start='2021-12-01 00:00:00'
end='2022-03-28 23:00:00'
min_horas = 5 #horas mínima de consumos para considerarlos
grupos = 5 #if empty the function search for the optimal number
bloque = 'Derechos' #**OJO**: Villabuena tiene que tener los datos C,B,A (está al revés) para que funcione bien
edificio='Edificio 1'
agregado=False
letras=3
portales=3
pisos=8  #derechos 8 villabuena 6
datos_sotanos = False
#smoothing for the temporal plot
smooth = [True, '3H']

# Creamos variable que diferencia bloques por pisos (los dos edificios)
bloques = np.concatenate([np.repeat('Derechos', 72), np.repeat('Villabuena', 54)])
bloques = pd.DataFrame(bloques).astype('category')

# En base al path, si es agregado y el año de inicio y de final unimos datos
#Estrcutrura ficheros
#consumos.csv
#temperatures.csv
#t_exterior.csv
#radiation.csv
complete, temp_complete, consumos, t_int, t_ext, radiation, m2_complete, m2= data_structure(path, agregado, start, end, bloques, bloque)

nombres = consumos.columns
nombres_complete = complete.columns

dates = pd.to_datetime(consumos.index, format='%d/%m/%Y %H:%M')
if agregado == True:
    year = 'agregado'
else:
    year = str(pd.to_datetime(start).year)

'''
Calculamos DT
'''
radiation = radiation.loc[:, 'rad']
diff = calculate_dt(t_ext, t_int)
diff_complete = calculate_dt(t_ext, temp_complete)



'''
Cosumos negativos forzados a ser 0
'''
consumos, out = check_data(consumos, 'consumption')
diff, out2 = check_data(diff, 'temp')

out_empty = np.union1d(out, out2)

diff = environment_analysis(diff, consumos, pisos, letras, portales,nombres)
diff_complete = environment_analysis(diff_complete, complete, pisos, letras, portales,nombres_complete)

diffT, diff = check_diff(diff, consumos)
diffT_complete, diff_complete = check_diff(diff_complete, complete)


var, var_con = calculate_kpi(consumos, diffT, m2)
var_complete, var_con_complete = calculate_kpi(complete, diffT_complete, m2_complete)

'''
Detección de días con un nivel de radiación por debajo de un nivel
'''
var, var_con, diff = detec_out_days(var, var_con, diff, radiation, t_ext.loc[:, 'temp'])

# Calculo de valores concretos para cada piso y para cada variable
horas = pd.DataFrame(var_con > 0).sum(axis=0)
horas_complete = pd.DataFrame(var_con_complete > 0).sum(axis=0)

diff_mean, var_sum, var_con_sum = acumulated(diff, var * 1000, var_con * 1000, nombres, horas)
diff_mean_complete, var_sum_complete, var_con_sum_complete = acumulated(diff_complete, var_complete * 1000,
                                                                        var_con_complete * 1000, nombres_complete,
                                                                        horas_complete)


####################################################################
matrix, df = create_matrix(var_sum, diff_mean)
matrix_complete, df_complete = create_matrix(var_sum_complete, diff_mean_complete)
matrix= environment(df, matrix,letras,pisos,portales)

# Delete empty dwellings
matrix, var_con_sum, nombres, horas = deletion(matrix, var_con_sum, nombres, horas, out_empty, datos_sotanos,portales,letras,pisos)

# Eliminamos del análisis pisos con muy pocas horas de consumo
matrix, var_con_sum, nombres = delete_dwellings_no_cons(horas, matrix, var_con_sum, nombres, min_horas)

# Eliminamos algun piso que no se pudo evitar que su entorno tuviera algún NaN
matrix = delete_missing_env(matrix, nombres)

df_piso = pd.DataFrame(matrix[:, np.array([0, 1])])
df_entorno = pd.DataFrame(matrix[:, np.array([3, 5, 7, 9])])

test_data(df_entorno, df_piso)

cluster = clustering(df_entorno, grupos)

lista = plot_environment(df_entorno, grupos, cluster, save_results, path, year, bloque)

kpi_group, kpi_red, kpi_green, t_group, t_red, t_green, Q_group, Q_red, Q_green, detection_sup, detection_inf = detection_imbalances(
    df_piso, var_con_sum, lista, nombres, path, year, save_results)

# Printeamos los pisos que forman cada grupos además de los pisos detectados en las posibles descompesaciones
info_detections(grupos, lista, nombres, detection_sup, detection_inf)

# Creamos un dataframe con los resultados: etiquetas de grupos, valores medio de grupos, valores medio por grupo de detección...
kpi_final = np.concatenate((kpi_group, kpi_red, kpi_green)).reshape(-1, 1)
temp_final = np.concatenate((t_group, t_red, t_green)).reshape(-1, 1)
Q_final = np.concatenate((Q_group, Q_red, Q_green)).reshape(-1, 1)

df_final = create_dataframe(kpi_final, temp_final, Q_final, grupos)
# Creamos gráficos donde junstamos la comparación del KPI y los saltos térmicos y los consumos específicos con los saltos térmicos
bar_line_plot(edificio, df_final, save_results, path, year)
