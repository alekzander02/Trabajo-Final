from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd

import datetime
from geopandas import read_file as gpd_read_file
import rioxarray
import pymannkendall as mk
from sklearn.linear_model import LinearRegression

shp_mapa = gpd.read_file("/content/drive/MyDrive/datos/Google_Colab_temp/shps/Departamentos.shp")
shp_mapa.head()
estaciones=pd.read_csv("/content/drive/MyDrive/datos/Google_Colab_temp/ema_xyz.csv", index_col=0)

fig, ax = plt.subplots(dpi=150)

im_shp = shp_mapa.geometry.boundary.plot(ax = ax, edgecolor = "black", linewidth = .5)
im_points = estaciones.plot.scatter(x="LON", y="LAT", ax = ax, c = "None", edgecolors = "blue", s = 5)

ax.set_ylabel("Latitud")
ax.set_xlabel("Longitud")