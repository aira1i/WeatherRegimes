###############################################################################
##################################LIBRARIES####################################
###############################################################################

# basic
import numpy as np
import xarray as xr
import pandas as pd

# data viasualization
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns
from IPython import display

# EOF analysis
from eofs.standard import Eof

# clustering
import sklearn
from sklearn.neighbors import NearestCentroid
from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from minisom import MiniSom

# ClimTool library
import climtools_lib as ctl
import climdiags as cd
from climtools_lib import *


###############################################################################
###################################DATASET#####################################
###############################################################################

# open dataset
dataDIR = "ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc"
ds = xr.open_dataset(dataDIR)

# convert longitude from 0-360 to -180 - +180
# necessary to work over EAT domain
ds = ds.assign_coords(lon = (ds.lon + 180) % 360 - 180).roll(lon = len(ds.lon)//2, roll_coords=True)

# remove the 29th of February of every leap year from the data set
# so every year has the same length
ds = ds.convert_calendar('noleap')

###############################################################################
##############################DERIVE ANOMALIES#################################
###############################################################################

# here we want to obtain geopotential height anomalies at 500hPa with the subtraction
# between the otiginal geopotential height data and the mean seasonal cycle

# seasonal cycle obtaine with a rolling mean of 20 days period
ds_roll   = ds.rolling(time=20,center=True).mean().groupby('time.dayofyear').mean('time')

# anomalies obtained by the subtraction with the seasonal cycle
anom_roll = ds.groupby('time.dayofyear') - ds_roll

# check if mean anomalies are zero
anom_roll_mean = anom_roll.mean("time")
zroll_mean = anom_roll_mean['z']
zroll_mean = zroll_mean[0,:,:]

#select only December-January-February (DJF) period and EAT spatioal domain
print(len(ds.time))
anomDJF = anom_roll.sel(time = ds.time.dt.month.isin([1, 2, 12]))
anomDJF = anomDJF.sel(lat = slice(30,87.5), lon = slice(-80,40))
zDJF = anomDJF['z']



###############################################################################
################################EOF CALCULATION################################
###############################################################################

#eof calculation

# convert xarray to numpy array, necessary to use this EOF calculator
# one may use the one implemented directly for xarray, but sometimes an internal
# conflict of the funtion arises with the version of xarray v2023.07.0
zDJF_np = zDJF.values

# calculation of the weights: we weight the area with the square root of the cosine
# of the latitude
coslat = np.cos(np.deg2rad(zDJF.coords['lat'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]

# compute the EOF and the PC weighted with the square root of the cosine of latitude
solver = Eof(zDJF_np, weights = wgts)
eof = solver.eofs(neofs=11)
pc  = solver.pcs(npcs=11, pcscaling=0)

#compute also the variance, the eigenvalues and the cumulative variance
varfrac = solver.varianceFraction()
lambdas = solver.eigenvalues()
cumulative = varfrac.cumsum()

#eliminate level (500hPa = 1) from the data array
eof = eof[:,0,:,:]


#################################################################################

# eof visualization: we put the plot here and not in the plot section because the
# only way to explore eof is via visualization, so it is necessary to give an
# immediate check

# conctourplot of the first 4 eof computed in the EAT domain
proj = ccrs.Orthographic(central_longitude=-30, central_latitude=60, globe=None)
proj_data = ccrs.PlateCarree() # Our data is lat-lon; thus its native projection is Plate Carree.
hfont = {'fontname':'Times New Roman'}
lats = zDJF.lat
lons = zDJF.lon

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), subplot_kw={'projection': proj})
axs = axs.flatten()

for i, ax in enumerate(axs):
    ax.coastlines(resolution='110m')
    ax.gridlines()
    c = ax.contourf(lons, lats, eof[i, :, :], levels=10, cmap='PuOr_r', transform=proj_data) 
    ax.set_title('EOF' + str(i + 1), fontsize=16, **hfont)
    
# adjust the colorbar
cb_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # [left, bottom, width, height]
cb = plt.colorbar(c, cax=cb_ax, orientation='horizontal')

# regulate manually the spaces between the plots
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.905, wspace=0, hspace=0.15)
plt.suptitle("EOF Analysis", fontsize=20, **hfont)
plt.show()
#plt.savefig("")

#################################################################################################

# just a quick visualization of the first 4 PCs timeseries

fig, axs = plt.subplots(2, 2 , figsize=(10, 7),)
axs[0, 0].plot(pc[:,0], color = 'C9')
axs[0, 0].set_title('PC1')
axs[0, 1].plot(pc[:,1], color = 'C11')
axs[0, 1].set_title('PC2')
axs[1, 0].plot(pc[:,2], color = 'C4')
axs[1, 0].set_title('PC3')
axs[1, 1].plot(pc[:,3], color = 'k')
axs[1, 1].set_title('PC4')

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='')

for ax in axs.flat:
    ax.label_outer()


###############################################################################
###############################CLUSTERING######################################
###############################################################################


# K-means custering

# random initialization of the centroids; 4 clusters selected
kms = KMeans(init='random', n_clusters=4, random_state=None) 
kms.fit(pc)
print(kms.cluster_centers_)
print(kms.labels_)


# here we compute the WR associationg the centroids derived from the K-means clustering
# to the first 4 eofs

w = range(0,4)

kms1 = kms.cluster_centers_[0,:4]
kms2 = kms.cluster_centers_[1,:4]
kms3 = kms.cluster_centers_[2,:4]
kms4 = kms.cluster_centers_[3,:4]

eof_lol = eof[w, :, :]

wr1 = np.zeros((24, 49), dtype = float)
wr2 = np.zeros((24, 49), dtype = float)
wr3 = np.zeros((24, 49), dtype = float)
wr4 = np.zeros((24, 49), dtype = float)


for w in range(0,4):
    wr1 += kms1[w]*eof_lol[w,:,:]

for w in range(0,4):
    wr2 += kms2[w]*eof_lol[w,:,:]

for w in range(0,4):
    wr3 += kms3[w]*eof_lol[w,:,:]

for w in range(0,4):
    wr4 += kms4[w]*eof_lol[w,:,:]


# here we find the 4 WR of the EAT domain given by the K-means clustering
wrKM = ([wr1, wr2, wr3, wr4])


# here calculate the frequency of occurrence of the 4 WR given by K-means
# given by the ratio between the number of days in each cluster and the total number of days
x = np.unique(kms.labels_, return_counts=True) #number of days in each cluster
y = len(kms.labels_)                           #total number of days

per = (x[1]*100)/y
per_show = (x[0], per)
print(per_show)


###############################################################################

# Gaussian Mixture model

# default options of the function are used because were the best for the present case
# but look at the documentation to change them; 4 clusters selected
gmm = GaussianMixture(n_components=4).fit(pc)
gm_labels = gmm.predict(pc)
gm_lables = gm_labels.tolist()
clfGM = NearestCentroid()
clfGM.fit(pc, gm_labels)
print(clfGM.centroids_)


# here we compute the WR associationg the centroids derived from the K-means clustering
# to the first 4 eofs

w = range(0,4)

gm1 = clfGM.centroids_[0,:4]
gm2 = clfGM.centroids_[1,:4]
gm3 = clfGM.centroids_[2,:4]
gm4 = clfGM.centroids_[3,:4]

eof_lol = eof[w, :, :]

wrgm1 = np.zeros((24, 49), dtype = float)
wrgm2 = np.zeros((24, 49), dtype = float)
wrgm3 = np.zeros((24, 49), dtype = float)
wrgm4 = np.zeros((24, 49), dtype = float)


for w in range(0,4):
    wrgm1 += gm1[w]*eof_lol[w,:,:]

for w in range(0,4):
    wrgm2 += gm2[w]*eof_lol[w,:,:]

for w in range(0,4):
    wrgm3 += gm3[w]*eof_lol[w,:,:]

for w in range(0,4):
    wrgm4 += gm4[w]*eof_lol[w,:,:]


# here we find the 4 WR of the EAT domain given by the K-means clustering
wrGM = ([wrgm1, wrgm2, wrgm3, wrgm4])


# here calculate the frequency of occurrence of the 4 WR given by Gaussian Mixture
# given by the ratio between the number of days in each cluster and the total number of days
xgm = np.unique(gm_labels, return_counts=True) #number of days for each cluster
ygm = len(gm_labels)                           #total number of days

pergm = (xgm[1]*100)/ygm
pergm_show = (xgm[0], pergm)
print(pergm)


#####################################################################################################

# Self-organizing map

som_shape = (1, 4)            #define the shape of the map
max_iter = 10000              #define the maximum number of iterations

#initialization: 11 input neurons (11 PCs); others are parameters that need to be tuned respect to the
# case study; random_seed is used to customize the start number of the random number generator
som = MiniSom(x = som_shape[0], y = som_shape[1], input_len = 11, sigma = 0.5, learning_rate = 0.8, neighborhood_function='gaussian', random_seed=17) 
som.train_batch(pc, 500, verbose=True)
# BMU
winner_coordinates = np.array([som.winner(x) for x in pc]).T
# with np.ravel_multi_index we convert the bidimensional coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
clf_som = NearestCentroid()
clf_som.fit(pc, cluster_index)   
print(clf_som.centroids_)

#obtain the position of the winner neuron respect to the training dataset
som.winner(pc[0])

# here we compute the WR associationg the centroids derived from the K-means clustering
# to the first 4 eofs

w = range(0,4)

sm1 = clf_som.centroids_[0, :4]
sm2 = clf_som.centroids_[1, :4]
sm3 = clf_som.centroids_[2, :4]
sm4 = clf_som.centroids_[3, :4]

eof_lol = eof[w, :, :]

wrs1 = np.zeros((24, 49), dtype = float)
wrs2 = np.zeros((24, 49), dtype = float)
wrs3 = np.zeros((24, 49), dtype = float)
wrs4 = np.zeros((24, 49), dtype = float)


for w in range(0,4):
    wrs1 += sm1[w]*eof_lol[w,:,:]

for w in range(0,4):
    wrs2 += sm2[w]*eof_lol[w,:,:]

for w in range(0,4):
    wrs3 += sm3[w]*eof_lol[w,:,:]

for w in range(0,4):
    wrs4 += sm4[w]*eof_lol[w,:,:]


# here we find the 4 WR of the EAT domain given by the K-means clustering
wrsSOM = ([wrs1, wrs2, wrs3, wrs4])


# here calculate the frequency of occurrence of the 4 WR given by Gaussian Mixture
# given by the ratio between the number of days in each cluster and the total number of days
xsom = np.unique(cluster_index, return_counts=True) #number of days for each cluster
ysom = len(cluster_index)                           #total number of days

persom = (xsom[1]*100)/ysom
persom_show = (xsom[0], persom)
persom_show


################################################################################

# Here we use a function taken from ClimTool library to match the PC sets relative
# to the different WR to facilitate the comparison between patterns in the plots and
# the frequencies of occurrence
# The Gaussian Mixture and SOM patterns are ordered taken the K-means ones as reference 

KM_GM  = match_pc_sets(wrKM, wrGM, latitude = None, verbose = False, bad_matching_rule = 'rms_mean_w_pos_patcor', matching_hierarchy = None)
KM_SOM = match_pc_sets(wrKM, wrsSOM, latitude = None, verbose = False, bad_matching_rule = 'rms_mean_w_pos_patcor', matching_hierarchy = None)
print(KM_GM)
print(KM_SOM)


###############################################################################
#################################PLOTS#########################################
###############################################################################

# contourplot of 500hPa geopotential height isolines at a certain time in the EAT domain

# select the day that one wants to plot and then take the geopotential height variable
# fromt the data array given by xarray
geocar = ds.sel(time=slice("2000-01-01", "2000-01-01"))
zcar1 = ds['z']
zcar = zcar1[0,0,:,:]       # Select the first element of the leading (time) dimension, and all lons and lats.

#make the contourplot
cLon = -30      #central longitude
cLat = 40       #central latitude
lonW = -120
lonE = 55
latS = -20
latN = 87.5
proj_map = ccrs.Orthographic(central_longitude=cLon, central_latitude=cLat)
proj_data = ccrs.PlateCarree() # Our data is lat,lon; thus its native projection is Plate Carree
res = '50m'
hfont = {'fontname':'Times New Roman'}
minVal = np.min(zcar1) 
maxVal = np.max(zcar1)
cint = 70
cintervals = np.arange(minVal, maxVal, cint)

fig = plt.figure(figsize=(13,9))
ax = plt.subplot(1,1,1,projection=proj_map)
ax.set_extent ([lonW,lonE,latS,latN])
ax.add_feature(cfeature.COASTLINE.with_scale(res))
#ax.add_feature(cfeature.STATES.with_scale(res))
ax.gridlines(draw_labels=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Longitude', labelpad = 10, fontsize=14,**hfont)
ax.set_ylabel('Latitude', rotation='vertical', va='center', labelpad = 10, fontsize=14, **hfont)
CL = ax.contour(zcar1.lon, zcar1.lat, zcar, cintervals, transform=proj_data, linewidths=1.5, colors='C11')
ax.clabel(CL, inline_spacing=0.4, fontsize=10, fmt='%.0f');

ax.set_title("Geopotential Height at $500 \, hPa$", fontsize=18, **hfont)

plt.tight_layout()
plt.show()
#plt.savefig('')

##################################################################################

#  time series plot of 500hPa geopotential height at a certain time in a certain transect

# period and transect selection
timeser00 = ds.sel(time=slice("1990-01-01", "2000-01-01"))
timeser0 = timeser00.sel(lat=50, lon=0)

# selection of the variable geopotential height from the xarray data array
z0 = timeser0['z']
z0 = z0[:,0]

#conversion from xarray to numpy array is convenient
z0 = z0.values

# make timeseries plot
start_date = datetime(1990, 1, 1)
end_date = datetime(1999, 12, 31)
time = np.array([start_date + timedelta(days=i) for i in range((end_date - start_date).days)])
hfont = {'fontname':'Times New Roman'}

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time, z0, linestyle='-', color='C11')
# time labels for x axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
ax.set_xlabel('Time [yrs]', fontsize=12, **hfont)
ax.set_ylabel('Geopotential Height [m]', fontsize=12, **hfont)
plt.title('Geopotential Height time series (10 years sample)', fontsize=16, **hfont)
#plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('')


##############################################################################################

# timeseries plot of the EAT seasonal cyvle given by rolling mean for a certain transect

# selection of the transect
seasroll = ds_roll.sel(lat=50, lon=0)

# selection of the geopotential height variable and conversion from xarray to numpy array
zsr = seasroll['z']
zsr = zsr[:,0]
zsr = zsr.values

# make the plot
start_date = datetime(1990, 1, 1)
end_date = datetime(1991, 1, 1)
time = np.array([start_date + timedelta(days=i) for i in range((end_date - start_date).days)])
hfont = {'fontname':'Times New Roman'}

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(time, zsr, linestyle='-', color='C4')
# time labels for x axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
ax.set_xlabel('Time [yrs]', fontsize=12, **hfont)
ax.set_ylabel('Geopotential Height [m]', fontsize=12, **hfont)
plt.title('Average seasonl cycle', fontsize=16, **hfont)
#plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('')


#############################################################################################

# time series of 500hPa geopotential height anomalies at a certain time at a certain transect

# period and transect selection
anomroll1 = anom_roll.sel(time=slice("1990-01-01", "2000-01-01"))
anomroll  = anomroll1.sel(lat=50, lon=0)

# selection of the variable geopotential height from the xarray data array
zar = anomroll['z']
zar = zar[:,0]

#conversion from xarray to numpy array is convenient
zar = zar.values

# make timeseries plot
start_date = datetime(1990, 1, 1)
end_date = datetime(1999, 12, 31)
time = np.array([start_date + timedelta(days=i) for i in range((end_date - start_date).days)])
hfont = {'fontname':'Times New Roman'}

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(time, zar, linestyle='-', color='C11')
# time labels for x axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

ax.set_xlabel('Time [yrs]', fontsize=12, **hfont)
ax.set_ylabel('Geopotential Height [m]', fontsize=12, **hfont)
plt.title('Geopotential Height Anomalies time series (10 years sample)', fontsize=16, **hfont)
#plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('')


###################################################################################################

#plot a contourplot of geopotential height anomalis for the EAT domain at a certain time

# selection of the day and then the geopotential height anomaly is taken from the xarray data array
geo_roll = anom_roll.sel(time=slice("2000-01-01", "2000-01-01"))
zroll1 = geo_roll['z']
zroll = zroll1[0,0,:,:] # Select the first element of the leading (time) dimension, and all lons and lats.

#make the contourplot

cLon = -30          #central lon
cLat = 40           #central lat
lonW = -120
lonE = 55
latS = -20
latN = 87.5
proj_map = ccrs.Orthographic(central_longitude=cLon, central_latitude=cLat)
proj_data = ccrs.PlateCarree() # Our data is lat-lon; thus its native projection is Plate Carree.
res = '50m'
hfont = {'fontname':'Times New Roman'}
minVal = np.min(zroll) 
maxVal = np.max(zroll)
cint = 70
cintervals = np.arange(minVal, maxVal, cint)

fig = plt.figure(figsize=(13,9))
ax = plt.subplot(1,1,1,projection=proj_map)
ax.set_extent ([lonW,lonE,latS,latN])
ax.add_feature(cfeature.COASTLINE.with_scale(res))
#ax.add_feature(cfeature.STATES.with_scale(res))
ax.gridlines(draw_labels=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Longitude', labelpad = 10, fontsize=14,**hfont)
ax.set_ylabel('Latitude', rotation='vertical', va='center', labelpad = 10, fontsize=14,**hfont)
CL = ax.contour(zcar1.lon, zcar1.lat, zroll, cintervals, transform=proj_data, linewidths=1.5, colors='C11')
ax.clabel(CL, inline_spacing=0.4, fontsize=10, fmt='%.0f');
ax.set_title("Geopotential Height Anomalies at $500 \, hPa$", fontsize=18, **hfont)
plt.tight_layout()
plt.show()
#plt.savefig('')


############################################################################################

# Scatterplot of the PCs
# here we have various options to plot:
# 1. Non-clusterd PCs
# 2. K-means clusterd PCs
# 3. Gaussian Mixture clustered PCs
# it is necessary to change the line "scatter" in the plot and change the title

x = pc[:, 0]
y = pc[:, 1]
z = pc[:, 2]
hfont = {'fontname':'Times New Roman'}

# create the scatterplot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# non-clusterd PCs
scatter = ax.scatter(x, y, z, color = 'C11', marker='o', s = 8)

# K-means clusterd PCs
#scatter = ax.scatter(x, y, z, c=kms.labels_, cmap='plasma', s = 8)

# Gaussian Mixture clustered PCs
#scatter = ax.scatter(x, y, z, c=gm_labels, cmap='inferno', s = 8)

# axes configuration
ax.set_xlabel('PC1', fontsize=12, **hfont)
ax.set_ylabel('PC2', fontsize=12, **hfont)
ax.set_zlabel('PC3', fontsize=12, **hfont)

ax.set_title("", fontsize=20, **hfont)
# rotate the plot for better visualization
ax.view_init(elev=15, azim=50)
plt.tight_layout()
plt.show()
#plt.savefig("")


###########################################################################################

# contour plot of the WR given by the 3 different clustering methods
# columns: clustering methods (k-means, gaussian mixture, SOM)
# raws: Circulation patterns (one needs to modify the plot following the order given by
# the ClimTool function match_pc_sets() for a correct comparison)

proj = ccrs.Orthographic(central_longitude=-40, central_latitude=60, globe=None)
lats = zDJF.lat
lons = zDJF.lon

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 20), subplot_kw={'projection': proj})

for i in range(4):
    for j in range(3):
        ax = axs[i, j]
        ax.coastlines()
        ax.gridlines(draw_labels=False)
      
c00 = axs[0,0].contourf(lons, lats, wrKM[3], cmap='PuOr_r', transform=ccrs.PlateCarree())
# Titolo specifico per la colonna di riferimento
axs[0, 0].set_title('K-means clustering \n \n $NAO+$', fontsize=20, **hfont)
c10 = axs[1,0].contourf(lons, lats, wrKM[2], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[1,0].set_title('$Sc. Blocking$', fontsize=20, **hfont)
c20 = axs[2,0].contourf(lons, lats, wrKM[0], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[2,0].set_title('$Atl. Ridge$', fontsize=20, **hfont)
c30 = axs[3,0].contourf(lons, lats, wrKM[1], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[3,0].set_title('$NAO-$', fontsize=20, **hfont)

c01 = axs[0,1].contourf(lons, lats, wrGM[2], cmap='PuOr_r', transform=ccrs.PlateCarree())
# Titolo specifico per la colonna di riferimento
axs[0, 1].set_title('Gaussian mixture model \n  \n $NAO+$', fontsize=20, **hfont)
c11 = axs[1,1].contourf(lons, lats, wrGM[1], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[1,1].set_title('$Sc. Blocking$', fontsize=20, **hfont)
c21 = axs[2,1].contourf(lons, lats, wrGM[3], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[2,1].set_title('$Atl. Ridge$', fontsize=20, **hfont)
c31 = axs[3,1].contourf(lons, lats, wrGM[0], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[3,1].set_title('$NAO-$', fontsize=20, **hfont)

c02 = axs[0,2].contourf(lons, lats, wrsSOM[0], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[0, 2].set_title('Self-organizing map 1 \n  \n $NAO+$', fontsize=20, **hfont)
c12 = axs[1,2].contourf(lons, lats, wrsSOM[3], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[1,2].set_title('$Sc. Blocking$', fontsize=20, **hfont)
c22 = axs[2,2].contourf(lons, lats, wrsSOM[1], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[2,2].set_title('$Atl. Ridge$', fontsize=20, **hfont)
c32 = axs[3,2].contourf(lons, lats, wrsSOM[2], cmap='PuOr_r', transform=ccrs.PlateCarree())
axs[3,2].set_title('$NAO-$', fontsize=20, **hfont)


# manually regulate the spaces between subplots
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.05, hspace=0.15)

# adjust the colorbar
cax = fig.add_axes([0.1, 0.05, 0.8, 0.015])  # [left, bottom, width, height]
cbar = plt.colorbar(c02, cax=cax, orientation='horizontal')
fig.suptitle("EAT Weather Regimes", fontsize=28, y=0.96, **hfont)
#plt.savefig("")
plt.show()

