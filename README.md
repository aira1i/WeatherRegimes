# Application of three clustering methods for the identification of Weather Regimes over the Euro-North Atlantic Region

Python code to compute Euro-North Atlantic Weather Regimes using ECMWF re-analysis ERA-40 daily mean geopotential height [m] at 500 hPa as input data (NetCDF format).
A useful dataset to test the code is available for ERA-5 re-analysis at https://climate.copernicus.eu/climate-reanalysis

The first step contists in the computation of the geopotential height anomalies [m], followed by the Empirical Orthogonal Functions (EOFs) decomposition to work with a reduced phase space.
We highlight that to open the dataset we use Xarray (https://docs.xarray.dev/en/stable/), useful to visualize the time varying geopotential height maps (longitude-latidute coordinates),
while to do the EOFs decomposition and the following steps we convert the data arrays in Numpy arrays because of a conflict in the eofs.xarray package for Xarray version v2023.07.0.
Alternatively one can choose to work directly with a different library to open the dataset (like NetCDF4) or to use a previous version of Xarray to avoid problems with eofs.xarray.

Then the clustering on the EOFs is done with three different clustering methods:
- K-means clustering, using sklearn.cluster.KMeans (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- Gaussian Mixture model, using sklearn.mixture.GaussianMixture (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- Self-organizing map, using minisom (https://github.com/JustGlowing/minisom)

After that we compute the Weather Regimes, associating the centroids of the 4 clusters to the calculated EOFs.
For the visualization of the regimes we use Cartopy package (https://pypi.org/project/Cartopy/) and the function match_pc_sets tanke from the ClimTool library 
(follow the steps explained in https://github.com/fedef17/ClimTools for the installation of the library and the needed environment).



