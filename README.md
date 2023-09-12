# Imbalances_detection
Scripts and data used for developing the study of imbalances detection through residential buildings. 


Folders to perform imbalance search:

- Scripts: 

	- imbalances_detections.py: main script. Selección de periodos para efectuar análisis, número de grupos ([] realizamos búsqueda de 	óptimo) y estructura del edificio

	- functions.py: specific functions to process the data and perform the detection with plots

	- decompesation_function.py: function to create the thermal environments
	
- Data:
	Files with data year by year and one agregate
	
- Graficos: 
	Folder to save the generated plots: barras, grupos y temporales
	

#IMBALANCE SEARCH

This search is based on 2 decompesation types:

- Type 1 (sup): High KPI (> p75) and low thermal gap (<p50) plus dwellings with very high KPI without having a high thermal gap (<p75)

- Type 2 (inf): Low KPI (< p25) and high thermal gap (>p50) plus dwellings with very low KPI without having a low thermal gap (>p25)