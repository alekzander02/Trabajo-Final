PISCOtx_m_Normals = PISCOtx.sel(time=slice("1981-01-31","2010-12-31")).groupby("time.month").mean("time")
PISCOtx_y_Normals = PISCOtx.mean("time")