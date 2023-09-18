import sys
import urllib.request
import numpy as np
import pandas as pd

#Specific case for GitHub repository
url1 = 'https://raw.githubusercontent.com/miguelmartinez95/Imbalances_detection/main/decompensation_function.py'
url2 = 'https://raw.githubusercontent.com/miguelmartinez95/Imbalances_detection/main/functions.py'

filename, headers = urllib.request.urlretrieve(url1, filename=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\Scripts\decompensation_function.py')
filename, headers = urllib.request.urlretrieve(url2, filename=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\Scripts\functions.py')

#Ejemplo de path (where the scripts are)
sys.path.insert(1, 'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection')
from decompensation_function import decompesation_analisis
from functions import data_structure

#Localización de datos
path=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\Data'

#Guardar gráficos
path2 = r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\Graficos'

#Años 2019 y 2020 con horas en 00:07:00, el resto con 00:00:00
start='2019-12-01 00:07:00'
end='2020-03-28 23:07:00'
min_horas = 5 #horas mínima de consumos para considerarlos
grupos = [] #if empty the function search for the optimal number
edificio = 'Villabuena'
agregado=False
letras=3
portales=3
pisos=6 #derechos 8 villabuena 6
datos_sotanos = False


# Creamos variable que diferencia bloques por pisos (los dos edificios)
bloques = np.concatenate([np.repeat('Derechos', 72), np.repeat('Villabuena', 54)])
bloques = pd.DataFrame(bloques).astype('category')

# En base al path, si es agregado y el año de inicio y de final unimos datos
consumos, t_int, t_ext, radiation = data_structure(path, agregado, start, end)
dates = pd.to_datetime(consumos.index, format='%d/%m/%Y %H:%M')
if agregado == True:
    year = 'agregado'
else:
    year = str(pd.to_datetime(start).year)

nombres = t_int.columns
t_out = t_ext.iloc[:, 4]
rad = radiation.iloc[:, 4]
m2 = pd.DataFrame(np.tile(np.array([84, 64, 84]), 42)) #metros cuadrados

#Elegimos el edificio concreto
consumos = consumos.iloc[:, np.where(bloques == edificio)[0]]
nombres = nombres[np.where(bloques == edificio)[0]]
t_int = t_int.iloc[:, np.where(bloques == edificio)[0]]
m2 = m2.iloc[np.where(bloques == edificio)[0]]

#smoothing for the temporal plot
smooth = [True, '3H']

decompesation_analisis(path2, edificio, consumos, t_int, t_out, rad, m2, min_horas, grupos,nombres, letras, portales, pisos, year, dates,smooth, datos_sotanos)
