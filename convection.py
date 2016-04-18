from lattice_boltzmann import LBM
from matplotlib import pyplot
import numpy

size = 100
state = LBM(size, 0.8)
state.boundry_stp[:, 0] = True
state.boundry_slip[:, -1] = True

density = 5.0
t = 10.0
vx = 0.0
vy = 0.0
R2 = state.space_ones
r = 3
state.state_space[:, :] = state.linearized_distribution(1.0*R2, 0.0*R2, 0.0*R2, 0.0*R2)#*(state.space_ones*numpy.linspace(25, 55, size)[:, numpy.newaxis]).T[:, :, numpy.newaxis]
#state.state_space[int(size/2)-5, int(size/2)-3] = state.linearized_distribution(density, vx, vy, t)
#state.state_space[int(size/2)+3, 0] = state.linearized_distribution(density, vx, vy, t)
#state.state_space[int(size/2)-r-20:int(size/2)+r-20, int(size/2)-r-20:int(size/2)+r-20] = state.linearized_distribution(density, vx, vy, t)
#state.plot_moment_0()
state.set_stp_distribution(0.5)
state.initalize()

"""state._plot_2d(state.boundry)
pyplot.title("boundry")
pyplot.show()
state._plot_2d(state.boundry_slip)
pyplot.title("boundry slip")
pyplot.show()
state._plot_2d(state.boundry_stp)
pyplot.title("boundry STP")
pyplot.show()
state._plot_2d(state.boundry_periodic)
pyplot.title("boundry periodic")
pyplot.show()

stuf1 = []
stuf2 = []
state.plot_moment_0()
state.plot_moment_1()
for i in range(100):
    for j in range(10000):
        if j%200 == 0:
            print(j)
        state.diffusion()
        state.mixing()
        b = numpy.sum(state.state_space, axis=0)
        stuf1.append(b[0, 0] + b[0, 1] + b[0, 2])
        stuf2.append(b[0, 6] + b[0, 7] + b[0, 8])

    state.plot_moment_0()
#    state.plot_moment_1_x()
#    state.plot_moment_1_y()
    state.plot_moment_1()
    pyplot.plot(numpy.sum(state.state_space, axis=2)[0, :])
    pyplot.show()
    pyplot.plot(stuf1)
    pyplot.plot(stuf2)
    pyplot.show()
#    state.plot_moment_2()
    print("saving data")
    state.save("HSE_atmosphere.dump")
    print("data saved")"""


state.load("HSE_atmosphere.dump")
#state.boundry_slip[:, 0] = True
state.initalize()
state.state_space[20, 0] = 0.9*state.state_space[20, 0]
space = state.state_space

density = 1.0
t = 50.0
th = 2
heating_range1 = (slice(None, None), slice(-th, None))
heating_range2 = (slice(int(size/4), -int(size/4)), slice(-th, None))
heating_range3 = (slice(None, None), slice(None, th))

def heating(self, heating_range, tf=1.04, d=1.0):
    M0 = numpy.sum(self.state_space[heating_range], axis=2) + d
    M1x = numpy.sum(self.state_space[heating_range]*(-self.dv*self.vxhat).flatten(), axis=2)
    M1y = numpy.sum(self.state_space[heating_range]*(-self.dv*self.vxhat).T.flatten(), axis=2)
    M2 = numpy.sum(self.state_space[heating_range]*self.dv**2*(self.vxhat**2 + self.vxhat.T**2).flatten(), axis=2)
    A = self.find_A(M0, M2)
    vx = self.find_vx(M0, M1x, M2)
    vy = self.find_vy(M0, M1y, M2)
    t = self.find_t(M0, M2) + tf
    self.state_space[heating_range] = self.state_space[heating_range] + self.mixing_rate*(self.linearized_distribution(A, vx, vy, t) - self.state_space[heating_range])

"""for j in range(2000):
    state.diffusion()
    state.mixing()
    heating(state, heating_range1)
    heating(state, heating_range2, 1.03)
#    state.state_space[:, -th:] = state.linearized_distribution(density, vx, vy, t1)
#    state.state_space[int(size/4):-int(size/4), -th:] = state.linearized_distribution(density, vx, vy, t2)
state.plot_moment_0(space)
state.plot_moment_1()
#state.plot_moment_2()
state.plot_vorticity()"""

"""for i in range(100):
    for j in range(10):
        state.diffusion()
        state.mixing()
#        heating(state, heating_range)
        heating(state, heating_range1)
        heating(state, heating_range2, 1.03)

    state.plot_moment_0(space)
#    state.plot_moment_1_x()
#    state.plot_moment_1_y()
    state.plot_moment_1()
#    state.plot_moment_2()
    state.plot_vorticity()"""


fig = pyplot.figure()
j = 0
sec = 1
image = pyplot.imshow(state.find_Moment1().T, animated=True, interpolation="none")

def mov(*args):
    global j
    j = j + 1
#    pyplot.clf()
    state.diffusion()
    state.mixing()
    heating(state, heating_range1, 10, 10)
    heating(state, heating_range2, 10, 1000)
    heating(state, heating_range3, -0.04, 1.004)
    print(j, "/ " + str(30*sec))
    image.set_array(state.find_Moment0().T)
#    image.set_array(numpy.sum(state.state_space - space,axis=2).T)
    return image, 
#    return pyplot.imshow(numpy.sum( state.state_space, axis=2))
#    return state.plot_moment_1(False)

from matplotlib import animation
ani = animation.FuncAnimation(fig, mov, interval=20, frames=30*sec, blit=True)
#ani.save("test.mp4", fps=30, bitrate=1600)
pyplot.show()
