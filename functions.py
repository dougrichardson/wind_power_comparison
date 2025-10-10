# functions for wind_power_comparison
import xarray as xr
import rioxarray
import numpy as np
import glob

def get_rez_boundary():
    """
    Return dict of lat/lon boundaries (slice objects) encompassing all REZs
    """
    lons = slice(133.5, 153.7)
    lats = slice(-43.4, -13.9)
    return {"lon": lons, "lat": lats}

def open_gwa_windspeed():
    """
    Return array of Global Wind Atlas wind speed over REZ boundary
    """
    rez_boundary = get_rez_boundary()
    
    gwa_ws = rioxarray.open_rasterio( #2008-2017 climatology
        "/g/data/ng72/ab4502/gwa/AUS_wind-speed_100m.tif",
    ).squeeze().rename(
        {"x": "lon", "y": "lat"}
    ).isel( # Reverse latitude coordinates
        lat=slice(None, None, -1)
    ).sel( # select REZ boundary region
        lat=rez_boundary["lat"],
        lon=rez_boundary["lon"]
    )
    return gwa_ws

def capacity_factor_vdW(W):
    """
    Computes capacity factor from wind speed data.
    Turbine power curve from van der Wiel et al (2019)
    https://www.sciencedirect.com/science/article/pii/S1364032119302862
    
    W: array, wind speed (m/s)
    """
    W_0 = 3.5 # cut-in speed (m/s)
    W_r = 13 # rated speed
    W_1 = 25 # cut-out speed (m/s)
    
    # Cubic
    c_f = (W ** 3 - W_0 ** 3) / (W_r ** 3 - W_0 ** 3)
    c_f = c_f.where(W >= W_0, 0) # Set values below cut-in to zero
    c_f = c_f.where(W < W_r, 1) # Set values above rated speed to 1
    c_f = c_f.where(W < W_1, 0) # Set values above cut-off to zero
    c_f = c_f.where(W.notnull(), np.nan) # Ensure NaNs are retained
    
    return c_f

def load_barra_c2_capacity_factor():
    """
    Returns Dataset of capacity factors for 2011-2023, 20min resolution
    """
    cf_paths = sorted(glob.glob("/scratch/w42/dr6273/BARRA-C2/derived/wind_capacity_factor/*.zarr"))
    cf_paths = cf_paths[32:-1] # select 2011-2013 files
    datasets = [xr.open_zarr(p, chunks={}) for p in cf_paths]
    return xr.combine_by_coords(datasets, compat="override", data_vars="minimal", coords="minimal")