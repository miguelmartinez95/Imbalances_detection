import sys
#Ejemplo de path
sys.path.insert(1, 'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Scripts')
from decompesation_function import decompesation_analisis

#Subir datos
path=r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Data'

#Guardar gráficos
path2 = r'E:\Documents\Doctorado\PAPERS\Paper_vivienda_social_Victoria\Paper\Graficos'

#Años 2019 y 2020 con horas en 00:07:00, el resto con 00:00:00
start='2019-12-01 00:07:00'
end='2020-03-28 23:07:00'
grupos = 5
edificio = 'Derechos'
agregado=False
decompesation_analisis(path,path2,agregado, start,end, grupos, edificio)
