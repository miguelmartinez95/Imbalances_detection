# Imbalances_detection
Scripts and data used for developing the study of imbalances detection through residential buildings.


Folder to perform imbalance search:

-Scripts: 
	imbalances_detections: main
	functions: two specific functions to process the data and perform the detection with plots
	decompesation_function: function to create the thermal environments
	
- Data:
	Files with data year by year and one agregate
	
- Graficos: 
	Folder to save the generated plots
	

#IMBALANCE SEARCH

This search is based on 4 decompesation types:

- Type 1 (sup): High KPI (> p75) and low thermal gap (<p50) plus dwellings with very high KPI without having a high thermal gap (<p75)

- Type 3 (inf): Low KPI (< p25) and high thermal gap (>p50) plus dwellings with very low KPI without having a low thermal gap (>p25)
