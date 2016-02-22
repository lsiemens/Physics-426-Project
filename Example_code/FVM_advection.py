#Simple advection of a density profile solved using the finite volume method.
from matplotlib import pyplot
import numpy

def vector(x, y):
    return numpy.array([x, y])

# find the value of F at the cell boundry
def extrapolate_x(F):
    Fx = numpy.zeros(shape=(F.shape[0], F.shape[1]+1))

    # interpolate values betwean cells
    Fx[:, 1:-1] = 0.5*(F[:, 1:] + F[:, :-1])

    # extrapolate values at boundry
    Fx[:, -1] = 1.5*F[:, -1] - 0.5*F[:, -2]
    Fx[:, 0] = 1.5*F[:, 0] - 0.5*F[:, 1]
    return Fx

# find the value of F at the cell boundry    
def extrapolate_y(F):
    Fy = numpy.zeros(shape=(F.shape[0]+1, F.shape[1]))

    # interpolate values betwean cells
    Fy[1:-1] = 0.5*(F[1:] + F[:-1])

    # extrapolate values at boundry
    Fy[-1] = 1.5*F[-1] - 0.5*F[-2]
    Fy[0] = 1.5*F[0] - 0.5*F[1]
    return Fy

# evolve the scalar field U for an "infitesimal" time dt    
def iteration(U, Fx, Fy, dt):
    # adding summ of flux through the boundry with area dx^2
    # for a time dt per volume dx^3
    # U_(n+1) = U_(n) + (dt/dx^3)*(summ of flux)*dx^2
    return U + (dt/dx)*(Fx[:, 1:]-Fx[:, :-1]+Fy[1:]-Fy[:-1])

x_min, x_max = -3.0, 3.0
resolution = 200
dx = (x_max - x_min)/resolution
dt = 0.2
#sub time steps
sub = 5

x = numpy.linspace(x_min, x_max, resolution)
y = numpy.linspace(x_min, x_max, resolution)
# uniform field
space = numpy.ones(shape=(resolution, resolution))

# Coordinate fields
Y, X = numpy.meshgrid(y, x)

# scalar field
rho = numpy.exp(-(X**2 + Y**2)/0.5**2)
# flow: simple advection
flow = vector(0.5*space, -1.0*space)
pyplot.imshow(rho, interpolation="none")
pyplot.show()

# simulate
for _ in range(100):
    for _ in range(sub):
        rho = iteration(rho, extrapolate_x(rho*flow[0]), extrapolate_y(rho*flow[1]), dt/sub)
    pyplot.imshow(rho, interpolation="none")
    pyplot.show()
