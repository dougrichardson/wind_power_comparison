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

def get_rolling_mean(compute, dataset, da, mask, rez, sampling_frequencies, lengths):
    """
    Return rolling mean arrays for desired sampling frequencies and window lengths

    compute: bool, whether to compute or read
    dataset: str, dataset and directory name e.g. 'BARRA-C2'
    da: array
    mask: array, REZ mask
    rez: str, name of REZ e.g. 'S6'
    sampling_frequencies: list, elements in format '20min' or 'Xhr', where X indicates every X-hourly time step
    lengths: list, elements integers indicating rolling length (in days)
    """
    def _get_skip(t):
        if t == "20min":
            nhrs = 0.33
            skip = None
            hr_time_steps = 24 * 3
        elif t[-2:] == "hr":
            nhrs = int(t[:-2])
            skip = nhrs * 3 # *3 because assumes 20min input data and 0000 first timestep
            hr_time_steps = 24 // nhrs # number of time steps per hour
        else:
            raise ValueError("Incorrect t specified")
        return nhrs, skip, hr_time_steps

    # Get REZ-mean data
    da = da.where(mask.sel(region=rez), drop=True)
    da = da.mean(["lat", "lon"])

    # for each sampling frequency get regional-average array and other variables
    sf_list = []
    roll_list = []
    nhrs_list = []
    for sf in sampling_frequencies:
        _nhr, _skip, _hr_ts = _get_skip(sf)
        
        sf_list.append(da.sel(time=slice(None, None, _skip)))
        roll_list.append(_hr_ts)
        nhrs_list.append(_nhr)

    print(roll_list)
    print(nhrs_list)
    return_list = []
    # For each window length...
    for l in lengths:
        
        len_list = []
        # For each sampling frequency, rolling window and hours-per-timestep
        for (da, roll, nhr) in zip(
            sf_list,
            roll_list,
            nhrs_list
        ):
            filename = "wind_capacity_factor_REZ_"+rez+"_"+str(nhr)+"hr_roll"+str(l)+"day_2011-2023"

            if compute:
                arr = da.chunk({"time": -1}).rolling(time=roll * l).mean()
                # arr = rolling_expand_dim(da, roll)
                arr = arr.to_dataset(name="cf_roll" + str(l) + "day")
                arr.to_netcdf(
                    "/g/data/ng72/dr6273/work/projects/wind_power_comparison/data/" + dataset + "/" + filename + ".nc"
                )
            else:
                arr = xr.open_mfdataset(
                    "/g/data/ng72/dr6273/work/projects/wind_power_comparison/data/" + dataset + "/" + filename + ".nc"
                )
                return_list.append(arr)

    if compute == False:
        
        return_dict = {}
        for i, sf in enumerate(sampling_frequencies):
            
            # Get arrays of a particular sampling
            len_arrays = [return_list[x] for x in range(i, len(return_list), len(sampling_frequencies))]
            
            # Get array from dataset and expand dim
            len_arrays = [
                da[list(da.data_vars)[0]].expand_dims(
                    {"window_len": [int(list(da.data_vars)[0][7:-3])]}
                ) for da in len_arrays
            ]
            
            concat_da = xr.concat(len_arrays, dim="window_len")
            return_dict[sf] = concat_da.compute()
            
        return return_dict