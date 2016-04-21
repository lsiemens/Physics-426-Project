from matplotlib import pyplot
from scipy.ndimage import imread
from scipy import fftpack
from radialfft import radial_fft
import numpy

r_clip = 0.85

image_flats = [("./data/CIMG2817.JPG", 1841, 1349, 1317),
("./data/exp2_p2/CIMG2927.JPG",1635, 1505, 1029),
("./data/exp2_p2/CIMG2928.JPG", 1677, 1393, 997)
#"./data/exp2_p2/CIMG2929.JPG",
#"./data/exp2_p2/CIMG2930.JPG",
#"./data/exp2_p2/CIMG2931.JPG",
#"./data/exp2_p2/CIMG2932.JPG",
#"./data/exp2_p2/CIMG2933.JPG",
#"./data/exp2_p2/CIMG2934.JPG"
]

image_data = [#("./data/CIMG2782.JPG", 1851, 1436, 1138, "r-", "heated water"),
              #("./data/CIMG2813.JPG", 1865, 1458, 881, "r-", None),
              #("./data/CIMG2815.JPG", 1936, 1662, 1054, "r-", None),
              ("./data/CIMG2816.JPG", 1793, 1444, 1271, "b-", "Non-rotating system"),
              ("./data/CIMG2817.JPG", 1841, 1349, 1317, "k-", "static fluid"),
              #("./data/CIMG2822.JPG", 1892, 1374, 1342, "g-", "fluid with sugar layer"),
              #("./data/CIMG2826.JPG", 1783, 1586, 1050, "g-", None),
              #("./data/CIMG2829.JPG", 1902, 1381, 1344, "g-", None),
              #("./data/CIMG2830.JPG", 1912, 1319, 1131, "g-", None),
              #("./data/CIMG2837.JPG", 1764, 1354, 1282, "b-", "boiling sugar layer"),
              #("./data/CIMG2838.JPG", 1824, 1373, 1308, "b-", None),
              ("./data/exp2_p1/CIMG2919.JPG", 1961, 1326, 785, "r-", "cooling rotating"),
              ("./data/exp2_p1/CIMG2920.JPG", 1889, 1391, 937, "r-", None),
              ("./data/exp2_p1/CIMG2921.JPG", 1945, 1426, 775, "r-", None),
              ("./data/exp2_p1/CIMG2922.JPG", 1847, 1528, 849, "r-", None)]

#image_data = image_data[:3]
fig1, ax1 = pyplot.subplots()
fig2, ax2 = pyplot.subplots()
fig3, ax3 = pyplot.subplots()
data_flats = [radial_fft(fname, x, y, r, r_clip=r_clip) for fname, x, y, r in image_flats]

data = [radial_fft(fname, x, y, r, r_clip=r_clip) for fname, x, y, r, style, label in image_data]

for i, d in enumerate(data[:2]):
    print("loaded " + str(i + 1) + " of " + str(len(image_data)) + ".")
    style, label = image_data[i][4], image_data[i][5]
#    d = radial_fft(fname, x, y, r, r_clip=r_clip)
    base = numpy.mean(numpy.array([data_flat.interpolate(d.x) for data_flat in data_flats]), axis=0)
    ax1.plot(d.x, d.data_powerspectrum + 1*d.data_powerspectrum_err, style)
    ax1.plot(d.x, d.data_powerspectrum, style, label=label)
    ax1.plot(d.x, d.data_powerspectrum - 1*d.data_powerspectrum_err, style)
    if label != "static fluid":
        ax2.plot(1.0/d.x, (d.data_powerspectrum + 1*d.data_powerspectrum_err)/base, style)
        ax2.plot(1.0/d.x, d.data_powerspectrum/base, style, label=label)
        ax2.plot(1.0/d.x, (d.data_powerspectrum - 1*d.data_powerspectrum_err)/base, style)
    else:
        ax2.plot(1.0/d.x, [1.0]*len(d.x), style, label=label)    
    ax3.plot(1.0/d.x, d.data_powerspectrum + d.data_powerspectrum_err, style)
    ax3.plot(1.0/d.x, d.data_powerspectrum - d.data_powerspectrum_err, style)
    ax3.plot(1.0/d.x, d.data_powerspectrum, style, label=label)

for i, d in enumerate([data[3]]):
    print("loaded " + str(i + 1) + " of " + str(len(image_data)) + ".")
    style, label = image_data[i][4], image_data[i][5]
#    d = radial_fft(fname, x, y, r, r_clip=r_clip)
    base = numpy.mean(numpy.array([data_flat.interpolate(d.x) for data_flat in data_flats]), axis=0)
    val = numpy.mean(numpy.array([f.interpolate(d.x) for f in data[2:]]), axis=0)
    val_err = numpy.mean(numpy.array([f.interpolate_err(d.x) for f in data[2:]]), axis=0)
#    ax1.plot(d.x, d.data_powerspectrum, style, label=label)
    ax1.plot(d.x, d.data_powerspectrum + 1*d.data_powerspectrum_err, style, label=label)
    ax1.plot(d.x, d.data_powerspectrum, style, label=label)
    ax1.plot(d.x, d.data_powerspectrum - 1*d.data_powerspectrum_err, style, label=label)
    if label != "static fluid":
        ax2.plot(1.0/d.x, (val + val_err)/base, "r-")
        ax2.plot(1.0/d.x, (val - val_err)/base, "r-")
        ax2.plot(1.0/d.x, val/base, "r-", label="Rotating system")
    else:
        ax2.plot(1.0/d.x, [1.0]*len(d.x), style, label=label)    
    ax3.plot(1.0/d.x, val + val_err, "r-")
    ax3.plot(1.0/d.x, val - val_err, "r-")
    ax3.plot(1.0/d.x, val, "r-", label="Rotating system")
ax1.legend()
ax1.set_xlabel("$\\nu$")
ax2.legend()
ax2.set_xlabel("$\\frac{\\lambda}{r}$")
ax2.set_ylabel("Normalized FFT radial profile")
ax3.legend()
ax3.set_xlabel("$\\frac{\\lambda}{r}$")
ax3.set_ylabel("FFT radial profile")
pyplot.show(fig1)
pyplot.show(fig2)
pyplot.show(fig3)
