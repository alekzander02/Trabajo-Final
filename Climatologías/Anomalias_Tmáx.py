anom_PISCOtx_m_Normals = [PISCOtx_m_Normals.sel(month = i) - PISCOtx_y_Normals for i in range(1,13)]
anom_PISCOtx_m_Normals = xr.concat(anom_PISCOtx_m_Normals, dim="month")
anom_PISCOtx_m_Normals.tx.attrs["long_name"] = "Tmax"
anom_PISCOtx_m_Normals["month"]= PISCOtx_m_Normals["month"]

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14,12))
for ((i, ax), month) in zip(enumerate(fig.axes), [1,2,3,4,5,6,7,8,9,10,11,12]):
    anom_PISCOtx_m_Normals.sel(month=month).tx.plot(ax=ax, cmap='Spectral_r', add_colorbar=True, extend='min')
    im_points = estaciones.plot.scatter(x="LON", y="LAT", ax = ax, c = "None", edgecolors = "black", s=2)
    shp_mapa.geometry.boundary.plot(ax = ax, edgecolor = "black", linewidth = .5)

for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis('tight')
    ax.set_xlabel('')

#Vs de anomalias
PISCOtx_monthly_anom = PISCOtx_monthly.groupby(PISCOtx_monthly.time.dt.strftime('%m')).apply(anomaly_from_mean)

tmin_monthly_tx_Peru = PISCOtx_monthly_anom.mean(dim=("latitude","longitude")).tx
tmin_yearly_tx_Peru = PISCOtx_monthly_anom.resample(time="1Y").mean(dim="time").mean(dim=("latitude","longitude")).tx
tmin_5monthly_tx_Peru = PISCOtx_monthly_anom.rolling(time=12, center=True).mean().mean(dim=("latitude","longitude")).tx

tmin_monthly_tx_Peru.plot(label = "Anomalía mensual", color="c")
tmin_yearly_tx_Peru.plot(label = "Anomalía anual", color="b")
tmin_5monthly_tx_Peru.plot(label = "Anomalía mensual (promedio movil 12 meses)", color="k")
plt.legend()

#Anomalias con tendencia
def xr_crop(shp_i, netcdf_i):
    # get box
    box_i = shp_i.total_bounds

    # crop based on box
    crop_netcdf_i = netcdf_i.where((netcdf_i["longitude"] > box_i[0]) &  # min lon
                                   (netcdf_i["longitude"] < box_i[2]) &  # max lon
                                   (netcdf_i["latitude"] > box_i[1]) &  # min lat
                                   (netcdf_i["latitude"] < box_i[3]),  # max lat
                                   drop=True)

    return crop_netcdf_i


#Rasterización.

def xr_shp_to_grid(shp_i, netcdf_array):
    # get real box
    shp_i_geometry = shp_i.geometry

    # adding crs
    mask = netcdf_array.rio.set_crs(shp_i.crs)

    # "rasterizing"
    mask = mask.rio.clip(shp_i_geometry, drop=False)

    # making "True/False" values
    mask.values[~np.isnan(mask.values)] = 1

    return mask.drop(["time", "spatial_ref"])


def xr_mask(grid_mask, netcdf_i):
    # masking
    mask_netcdf_i = netcdf_i.where(grid_mask == True)

    return mask_netcdf_i

PISCOtx_anom_crop = xr_crop(shp_i=shp_mapa, netcdf_i=PISCOtx_monthly_anom)

shp_exp_grid = xr_shp_to_grid(shp_i = shp_mapa,
                              netcdf_array = PISCOtx_anom_crop.tx.isel(time=15))

PISCOtx_anom_masked = xr_mask(grid_mask = shp_exp_grid,
                             netcdf_i = PISCOtx_anom_crop)

PISCOtx_anom_masked

array_numpy_tx = PISCOtx_anom_masked.tx.to_numpy()

xr.apply_ufunc(np.mean,
               PISCOtx_anom_masked,
               input_core_dims=[["latitude","longitude"]],
               vectorize=True)

xr.apply_ufunc(np.mean,
               PISCOtx_anom_masked,
               input_core_dims=[["time"]],
               vectorize=True)

#Seleccionamos un mes especifico para hacer la serie interanual
PISCOtx_anom_masked_01 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==1).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_02 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==2).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_03 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==3).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_04 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==4).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_05 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==5).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_06 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==6).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_07 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==7).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_08 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==8).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_09 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==9).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_10 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==10).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_11 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==11).mean(dim=["latitude","longitude"]).to_dataframe()
PISCOtx_anom_masked_12 = PISCOtx_anom_masked.sel(time=PISCOtx_anom_masked.time.dt.month==12).mean(dim=["latitude","longitude"]).to_dataframe()

#enero
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_01.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Enero", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#febrero
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_02.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Febrero", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#marzo
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_03.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Marzo", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#abril
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_04.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Abril", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#mayo
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_05.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Mayo", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#junio
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_06.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Junio", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#julio
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_07.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Julio", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#Agosto
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_08.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Agosto", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#Septiembre
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_09.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Septiembre", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#Octubre
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_10.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Octubre", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#Noviembre
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_11.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Noviembre", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()

#Diciembre
fig, ax = plt.subplots(figsize=(6, 4), dpi=80, sharey=True)

sct_data = pd.concat([pd.Series(PISCOtx_anom_masked_12.iloc[:, 0].values), pd.Series(np.arange(1981, 2017))], axis=1)
X = sct_data.iloc[:, 1].values.reshape(-1, 1)
Y = sct_data.iloc[:, 0].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
f = lambda x: linear_regressor.coef_ * x + linear_regressor.intercept_
x = np.array([min(X), max(X)])

ax.scatter(X, Y, marker='+', c="blue", linewidths=2)
ax.plot(x, f(x), c="black")
ax.set_title("Anomalía mensual (Tmáx) - Diciembre", size=15)
ax.annotate(np.round(float(linear_regressor.coef_[0]), 2) * 10, (1985, 0.75),
            size=9.5)  # Cambio de temperatura a escala decadal (anual*10 años)

plt.tight_layout()