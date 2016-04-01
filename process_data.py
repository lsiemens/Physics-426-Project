from matplotlib import pyplot
from scipy.ndimage import imread
from scipy import fftpack
from radial_fft import radial_fft
import numpy

r_scale = 1.0

image_data = [("./data/CIMG2782.JPG", 1851, 1436, 1138, "r-", "heated water"),
              ("./data/CIMG2813.JPG", 1865, 1458, 881, "r-", None),
              ("./data/CIMG2815.JPG", 1936, 1662, 1054, "r-", None),
              ("./data/CIMG2816.JPG", 1793, 1444, 1271, "r--", "cooling water"),
              ("./data/CIMG2817.JPG", 1841, 1349, 1317, "k-", "static fluid"),
              ("./data/CIMG2822.JPG", 1892, 1374, 1342, "g-", "fluid with sugar layer"),
              ("./data/CIMG2826.JPG", 1783, 1586, 1050, "g-", None),
              ("./data/CIMG2829.JPG", 1902, 1381, 1344, "g-", None),
              ("./data/CIMG2830.JPG", 1912, 1319, 1131, "g-", None),
              ("./data/CIMG2837.JPG", 1764, 1354, 1282, "b-", "boiling sugar layer"),
              ("./data/CIMG2838.JPG", 1824, 1373, 1308, "b-", None)]

#image_data = image_data[:5]
fig1, ax1 = pyplot.subplots()
fig2, ax2 = pyplot.subplots()

base = radial_fft("./data/CIMG2817.JPG", 1841, 1349, 1317, r_scale=r_scale)


for i, (fname, x, y, r, style, label) in enumerate(image_data):
    print("loaded " + str(i + 1) + " of " + str(len(image_data)) + ".")
    d = radial_fft(fname, x, y, r, r_scale=r_scale)
    ax1.plot(d.x, d.data_powerspectrum, style, label=label)
    ax2.plot(1.0/d.x, d.data_powerspectrum/base.interpolate(d.x), style, label=label)
ax1.legend()
ax1.set_xlabel("$\\nu$")
ax2.legend()
ax2.set_xlabel("$\\lambda$")
pyplot.show(fig1)
pyplot.show(fig2)
