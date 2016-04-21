from matplotlib import pyplot
from scipy.ndimage import imread
from scipy import interpolate
from scipy import fftpack
import numpy

def radial_profile(data, center):
    y, x = numpy.indices((data.shape))
    r = numpy.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(numpy.int)
    value_bin = numpy.bincount(r.ravel(), data.ravel())
    value_sqr_bin = numpy.bincount(r.ravel(), (data**2).ravel())
    num_bin = numpy.bincount(r.ravel())
#    print(value_bin, num_bin)
#    return value_bin/num_bin
    return value_bin/num_bin, numpy.sqrt(value_sqr_bin/num_bin - (value_bin/num_bin)**2)/numpy.sqrt(num_bin)

class radial_fft:
    def __init__(self, fname, x, y, r, r_clip=1.0, r_ramp=5, r_size=1.0, max_input=255.0):
        self.x = x
        self.y = y
        self.r = r
        self.r_clip = r_clip #cliping radii
        self.r_ramp = r_ramp #ramp from background to mask
        self.r_size = r_size #radii size
        self.max_input = max_input
        if isinstance(fname, str):
            self.fname = fname
            self.raw_data = imread(fname)
        else:
            self.fname = None
            self.raw_data = fname
        self.data = None
        self._interpolate_fft = None
        self._interpolate_fft_err = None
        
        self.f_sampling = None
        
        self._mask_data()
        self._fft()

    def _mask_data(self):
        #scale r (remove error on side)
        r = int(self.r_clip*self.r)
        X, Y = numpy.meshgrid(numpy.linspace(0, self.raw_data.shape[1], self.raw_data.shape[1]), numpy.linspace(0, self.raw_data.shape[0], self.raw_data.shape[0]))
        #define mask
        mask = 1.0 - 0.5*(numpy.tanh((numpy.sqrt((X - self.x)**2 + (Y - self.y)**2) - r)/self.r_ramp) + 1.0)
        #find luminance
        if self.fname != None:
            luminance = numpy.sum((self.raw_data[:, :]/self.max_input)**2, axis=2)/3.0
        else:
            if len(self.raw_data.shape) > 2:
                luminance = numpy.sum((self.raw_data[:, :]/self.max_input)**2, axis=2)/3.0
            else:
                luminance = numpy.abs(self.raw_data[:, :]/self.max_input)
        #mask data
        luminance = luminance*mask
        #normalze
        mean = numpy.average(luminance, weights=mask)
#        luminance = luminance/mean - mask
#        luminance = luminance/numpy.max(numpy.abs(luminance))
        luminance = luminance - mean*mask
#        luminance = luminance/numpy.max(numpy.abs(luminance))
        luminance = luminance[self.y-r-int(self.r_ramp):self.y+r+int(self.r_ramp), self.x-r-int(self.r_ramp):self.x+r+int(self.r_ramp)]
        self.data = luminance
        
    def _fft(self):
        r = int(self.r_clip*self.r)
        fft = fftpack.fft2(self.data)
        fft = fftpack.fftshift(fft)
        
        self.f_max = 2.0*self.r/(4.0*self.r_size)
        self.f_min = 1.0/(2.0*self.r_size)
        
        powerspec = numpy.abs(fft)**2
        self.data_powerspectrum, self.data_powerspectrum_err = radial_profile(numpy.log10(powerspec), (powerspec.shape[0]/2, powerspec.shape[0]/2))
        
        self.x = numpy.linspace(self.f_min, len(self.data_powerspectrum)*self.f_min, len(self.data_powerspectrum))

        self._interpolate_fft = interpolate.UnivariateSpline(self.x, self.data_powerspectrum, s=0, k=2)
        self._interpolate_fft_err = interpolate.UnivariateSpline(self.x, self.data_powerspectrum_err, s=0, k=2)
    
    def interpolate(self, x):
        return self._interpolate_fft(x)

    def interpolate_err(self, x):
        return self._interpolate_fft_err(x)

""" 
d = radial_fft("./data/CIMG2817.JPG", 1841, 1349, 1317)
import voronoi
import random
random.seed()
bw = voronoi.bowyer_watson(2.0)
for _ in range(10):
    bw.add_point(voronoi.vec(random.uniform(-1, 1), random.uniform(-1, 1)))
v = bw.get_voronoi(2000)
v.gaussian(0.04, 0.1)
data = 1.0 - 0.1*v.raster
d = radial_fft(data, 1000, 1000, 990, max_input=1.0)
"""
