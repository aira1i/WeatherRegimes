# WeatherRegimes
Python code to compute Euro-North Atlantic Weather Regimes using ECMWF re-analysis ERA-40 daily mean geopotential height [m] at 500 hPa as input data (NetCDF format).
A useful dataset to test the code is available for ERA-5 re-analysis at https://climate.copernicus.eu/climate-reanalysis

The first step contists in the computation of the geopotential height anomalies [m], followed by the EOF decomposition to work with a reduced phase space.
We highlight that to open the dataset we use Xarray (https://docs.xarray.dev/en/stable/), useful to visualize the time varying geopotential height maps (longitude-latidute coordinates),
while to do the EOF decomposition and the following steps we convert the data arrays in Numpy arrays because a conflict in the eofs.xarray package for Xarray version v2023.07.0.



