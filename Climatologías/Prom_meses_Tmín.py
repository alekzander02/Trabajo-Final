PISCOtn = xr.open_dataset("/content/drive/MyDrive/TPM_II/datos/PISCO_temperature/tn/PISCOdtn_v1.1.nc")

PISCOtn_yearly = PISCOtn.resample(time="1Y").mean(dim="time")
PISCOtn_monthly = PISCOtn.resample(time="1M").mean(dim="time")

PISCOtn_time_y = PISCOtn_yearly.sel(time=slice("1981-01-31","2010-12-31"))
PISCOtn_time_m = PISCOtn_monthly.sel(time=slice("1981-01-31","2010-12-31"))

PISCOtn_m_Normals = PISCOtn.sel(time=slice("1981-01-31","2010-12-31")).groupby("time.month").mean("time")
PISCOtn_y_Normals = PISCOtn.mean("time")

#Vs climatologias
PISCOtn_monthly.sel(time=slice("1981-01-31","2010-12-31")).mean(dim=("latitude","longitude")).tn.plot()
PISCOtn_yearly.sel(time=slice("1981-01-01", "2010-12-31")).mean(dim=("latitude","longitude")).tn.plot(marker="o")

#Prom en meses
PISCOtn_m_Normals.tn.attrs["long_name"] = "Tmin"

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14,12))
for ((i, ax), month) in zip(enumerate(fig.axes), [1,2,3,4,5,6,7,8,9,10,11,12]):
    PISCOtn_m_Normals.sel(month=month).tn.plot(ax=ax, cmap='Spectral', add_colorbar=True, extend='max')
    im_shp = shp_mapa.geometry.boundary.plot(ax = ax, edgecolor = "black", linewidth = .5)
    im_points = estaciones.plot.scatter(x="LON", y="LAT", ax = ax, c = "None", edgecolors = "black", s = 5)
for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis('tight')
    ax.set_xlabel('')