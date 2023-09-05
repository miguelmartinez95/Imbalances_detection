import sys
import urllib.request
import numpy as np
import pandas as pd

url1 = 'https://raw.githubusercontent.com/miguelmartinez95/Imbalances_detection/main/decompensation_function_def.py'
url2 = 'https://raw.githubusercontent.com/miguelmartinez95/Imbalances_detection/main/functions_def.py'

filename, headers = urllib.request.urlretrieve(url1, filename=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\decompensation_function_def.py')
filename, headers = urllib.request.urlretrieve(url2, filename=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection\functions_def.py')

#Ejemplo de path
sys.path.insert(1, 'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Imbalance_detection')
from decompensation_function_def import decompesation_analisis
from functions_def import data_structure

#Subir datos
path=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Data'

#Guardar gráficos
path2 = r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Graficos'

#Años 2019 y 2020 con horas en 00:07:00, el resto con 00:00:00
start='2021-12-01 00:00:00'
end='2022-03-28 23:00:00'
grupos = 5
edificio = 'Derechos'
agregado=False
letras=3
portales=3
pisos=8

# Creamos variable que diferencia bloques por pisos (hasta ibamos con los dos bloques)
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

#elegimos el edificio concreto
consumos = consumos.iloc[:, np.where(bloques == edificio)[0]]
#t_out = t_out.iloc[:, np.where(bloques == edificio)[0]]
nombres = nombres[np.where(bloques == edificio)[0]]
t_int = t_int.iloc[:, np.where(bloques == edificio)[0]]
#rad = rad.iloc[:, np.where(bloques == edificio)[0]]

#Metros cuadrados
m2 = pd.DataFrame(np.tile(np.array([84, 64, 84]), 24))
#m2 = pd.DataFrame(np.tile(np.array([84, 64, 84]), 18))

decompesation_analisis(path2, consumos, t_int, t_out, rad, m2, grupos,nombres, letras, portales, pisos, year, dates)
