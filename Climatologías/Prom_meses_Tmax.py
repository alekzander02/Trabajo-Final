PISCOtx = xr.open_dataset("/content/drive/MyDrive/TPM_II/datos/PISCO_temperature/tx/PISCOdtx_v1.1.nc")

PISCOtx_yearly = PISCOtx.resample(time="1Y").mean(dim="time")
PISCOtx_monthly = PISCOtx.resample(time="1M").mean(dim="time")

PISCOtx_time_y = PISCOtx_yearly.sel(time=slice("1981-01-31","2010-12-31"))
PISCOtx_time_m = PISCOtx_monthly.sel(time=slice("1981-01-31","2010-12-31"))

PISCOtx_m_Normals = PISCOtx.sel(time=slice("1981-01-31","2010-12-31")).groupby("time.month").mean("time")
PISCOtx_y_Normals = PISCOtx.mean("time")

PISCOtx_m_Normals.tx.attrs["long_name"] = "Tmax"
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
for ((i, ax), month) in zip(enumerate(fig.axes), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    PISCOtx_m_Normals.sel(month=month).tx.plot(ax=ax, cmap='Spectral', add_colorbar=True, extend='max')
    im_shp = shp_mapa.geometry.boundary.plot(ax=ax, edgecolor="black", linewidth=.5)
    im_points = estaciones.plot.scatter(x="LON", y="LAT", ax=ax, c="None", edgecolors="black", s=5)

for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis('tight')
    ax.set_xlabel('')

plt.tight_layout()

a=PISCOtx_time_m.sel(time=slice("1981-01-01", "2010-12-31")).mean(dim=("latitude","longitude")).tx
b=PISCOtx_time_y.sel(time=slice("1981-01-01", "2010-12-31")).mean(dim=("latitude","longitude")).tx

a.plot(label="Temperatura Mensual")
b.plot(label="Temperatura Anual",marker="o")
plt.title("Promedio de Temperatura Mensual vs Anual [1981-2010]")
