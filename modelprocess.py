from matplotlib import pyplot
from scipy.ndimage import imread
from scipy import fftpack
from radialfft import radial_fft
import numpy
import voronoi
try:
    import cPickle as pickle
except ImportError:
    print("Failed to load cPickle")
    import pickle
    
r_clip = 0.85
import random
random.seed()
res = 2000

num_points = 40

data = []
for num_points in range(10, 170, 10):    
    distance = []
    distance_err = []
    fft_data = []
    for _ in range(10):
        bw = voronoi.bowyer_watson(2.0)
        for _ in range(num_points):
            bw.add_point(voronoi.vec(random.uniform(-1, 1), random.uniform(-1, 1)))
        v = bw.get_voronoi(res)
        dist, dist_err = v.mean_distance(300)
        distance += [dist]
        distance_err += [dist_err]
        v.gaussian(1/32., 1.5/16., resolution=res)
        rad_fft = radial_fft(1.0 - 0.1*v.raster, res/2, res/2, res/2, r_clip=r_clip, max_input=1.0)
        fft_data += [(rad_fft.interpolate, rad_fft.interpolate_err)]
        rad_fft.fname, rad_fft.raw_data, rad_fft.data = None, None, None
    
    data += [(rad_fft.x, numpy.mean(distance), numpy.mean(distance_err), fft_data)]

with open("model.dump", "wb") as fout:
    pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

""" 
data = None
with open("test1.dump", "rb") as fin:
    data = pickle.load(fin)

print(data[1], data[2])
pyplot.plot(data[0], data[3][-1][0](data[0]))
pyplot.show()
"""
