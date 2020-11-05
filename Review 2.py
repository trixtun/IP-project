import mahotas as mh 
import numpy as np
from matplotlib import pyplot as plt
from IPython.html.widgets import interact

plt.rcParams['figure.figsize'] = (10.0, 8.0) # 10 x 8inches 
plt.gray()
dna =mh.demos.load('Nuclear')
print(dna.shape) 
dna=dna.max(axis=2) 
print(dna.shape) 
plt.imshow(dna)
T_otsu = mh.otsu(dna) 
print(T_otsu) 
plt.imshow(dna > T_otsu)
T_mean = dna.mean() 
print(T_mean)
plt.imshow(dna > T_mean)
dnaf = mh.gaussian_filter(dna, 2.) 
T_mean = dnaf.mean()

bin_image = dnaf > T_mean 
plt.imshow(bin_image)
labeled, nr_objects = mh.label(bin_image) 
print(nr_objects)
plt.imshow(labeled) 
plt.jet()
@interact(sigma=(1.,16.)) 
def check_sigma(sigma):
    dnaf = mh.gaussian_filter(dna.astype(float), sigma) 
    maxima = mh.regmax(mh.stretch(dnaf))
    maxima = mh.dilate(maxima, np.ones((5,5))) 
    plt.imshow(mh.as_rgb(np.maximum(255*maxima, dnaf), dnaf, dna > T_mean))
sigma = 12.0

dnaf = mh.gaussian_filter(dna.astype(float),sigma) 
maxima = mh.regmax(mh.stretch(dnaf)) 
maxima,_= mh.label(maxima) 
plt.imshow(maxima)
dist = mh.distance(bin_image) 
plt.imshow(dist)
dist = 255 - mh.stretch(dist)
watershed = mh.cwatershed(dist,maxima) 
plt.imshow(watershed)
watershed *= bin_image 
plt.imshow(watershed)
watershed = mh.labeled.remove_bordering(watershed) 
plt.imshow(watershed)
sizes = mh.labeled.labeled_size(watershed)
# The conversion below is not necessary in newer versions of mahotas: watershed = watershed.astype(np.intc)
@interact(min_size=(100,4000,20)) 
def do_plot(min_size):
    filtered = mh.labeled.remove_regions_where(watershed, sizes < min_size) 
    print("filtering {}...".format(min_size))
    plt.imshow(filtered) 
min_size = 2000
filtered = mh.labeled.remove_regions_where(watershed, sizes < min_size)
labeled,nr_objects = mh.labeled.relabel(filtered) 
print("Number of cells: {}".format(nr_objects))
