from matplotlib import pyplot
import numpy

#Relaxed velocity profile
# 1/36 1/9 1/36   1/9 4/9 1/9   1/36 1/9 1/36

class LBM:
    def __init__(self, resolution, mixing_rate=0.1):
        self.resolution = resolution
        self.mixing_rate = mixing_rate
        self.interpolation=None#"none"
        self.colorbar = pyplot.cm.gray
        self.conserve_first_moment = True

        self.dx = 1.0 # in m
        self.m = 1.0 # in kg
        self.dv = 1.0 # in m/s
        self.t_0 = 200.0 # in K
        self.k = 0.001
        
        ##---START Precompute---##
        
        self.e_0 = numpy.exp(-self.m*self.dv**2/(2*self.k*self.t_0))

        self.a_0 = -(1+2*self.e_0)/(self.dv**2*(1+4*self.e_0+4*self.e_0**2))
        self.a_1 = (1+4*self.e_0)/(1+4*self.e_0+4*self.e_0**2)

        self.b_0 = self.k*self.t_0*(1/self.e_0+4+4*self.e_0)/(2*self.m)
        self.b_1 = -(1+4*self.e_0+4*self.e_0**2)
        self.b_2 = self.dv**2*(1 + 6*self.e_0 + 8*self.e_0**2)

        self.c_0=-(self.k*self.t_0**2)/(2*self.m*self.dv**2)
        self.c_1 = -(1/self.e_0 + 12 + 60*self.e_0 + 160*self.e_0**2 + 240*self.e_0**3 + 192*self.e_0**4 + 64*self.e_0**5)
        self.c_2 = 4*self.dv**2*(1 + 10*self.e_0 + 40*self.e_0**2 + 80*self.e_0**3 + 80*self.e_0**4 + 32*self.e_0**5)
        self.c_3 = -(1 + 10*self.e_0 + 40*self.e_0**2 + 80*self.e_0**3 + 80*self.e_0**4 + 32*self.e_0**5)
        self.c_4 = self.dv**2*(1 + 12*self.e_0 + 56*self.e_0**2 + 128*self.e_0**3 + 144*self.e_0**4 + 64*self.e_0**5)
        
        ##---END Precompute---##
        
        self.velocity = numpy.linspace(-self.dv, self.dv, 3)
        self._distribution_static = numpy.array([self.e_0**2, self.e_0, self.e_0**2, self.e_0, 1.0, self.e_0, self.e_0**2, self.e_0, self.e_0**2])
        self.state_space = numpy.zeros(shape=(self.resolution, self.resolution, 9))
        self.const_2space = numpy.ones(shape=(self.resolution, self.resolution))

        self.vxhat = numpy.linspace(-1.0, 1.0, 3)*numpy.ones(shape=(3))[:, numpy.newaxis]
        
        if (mixing_rate < 0.0) or (mixing_rate > 1.0):
            raise ValueError("Invalid mixing rate (" + str(mixing_rate) + "). Mixing must be within [0.0, 1.0].")
            
    def find_A(self, Moment0, Moment2):
        return self.a_0*Moment2 + self.a_1*Moment0
    
    def find_vx(self, Moment0, Moment1x, Moment2):
        return self.b_0*Moment1x/(Moment2*self.b_1 + self.b_2*Moment0)
    
    def find_vy(self, Moment0, Moment1y, Moment2):
        return self.b_0*Moment1y/(Moment2*self.b_1 + self.b_2*Moment0)
    
    def find_t(self, Moment0, Moment2):
        return self.c_0*(self.c_1*Moment2 + self.c_2*Moment0)/(self.c_3*Moment2 + self.c_4*Moment0)
    
    def _plot_2d(self, data, vmin=None, vmax=None):
        pyplot.imshow(data.T, interpolation=self.interpolation, cmap=self.colorbar, vmin=vmin, vmax=vmax)
        pyplot.colorbar()
        
    def plot_moment_0(self):
        self._plot_2d(numpy.sum(self.state_space, axis=2), 0.0)
        pyplot.title("Zeroeth Moment")
        pyplot.show()
        
    def plot_moment_1_x(self):
        self._plot_2d(numpy.sum(self.state_space*(-self.dv*self.vxhat).flatten(), axis=2))
        pyplot.title("First Moment X")
        pyplot.show()
        
    def plot_moment_1_y(self):
        self._plot_2d(numpy.sum(self.state_space*(-self.dv*self.vxhat).T.flatten(), axis=2))
        pyplot.title("First Moment Y")
        pyplot.show()

    def plot_moment_1(self, norm=False, vmax=None):
        if norm:
            self._plot_2d(numpy.sqrt(numpy.sum(self.state_space*(-self.dv*self.vxhat).flatten(), axis=2)**2 + numpy.sum(self.state_space*(-self.dv*self.vxhat.T).flatten(), axis=2)**2)/numpy.sum(self.state_space, axis=2), vmax=vmax)
        else:
            self._plot_2d(numpy.sqrt(numpy.sum(self.state_space*(-self.dv*self.vxhat).flatten(), axis=2)**2 + numpy.sum(self.state_space*(-self.dv*self.vxhat.T).flatten(), axis=2)**2), vmax=vmax)
        pyplot.title("First Moment")
#        pyplot.show()
        
    def plot_moment_2(self):
        self._plot_2d(numpy.sum(self.state_space*self.dv**2*(self.vxhat**2 + self.vxhat.T**2).flatten(), axis=2), 0.0)
        pyplot.title("Second Moment")
        pyplot.show()
    
    def plot_vorticity(self):
        x_speed = numpy.sum(self.state_space*(self.dv*self.vxhat).flatten(), axis=2)
        x_speed = (x_speed[:-1, 1:] - x_speed[:-1, :-1])/self.dx
        y_speed = numpy.sum(self.state_space*(self.dv*self.vxhat).T.flatten(), axis=2)
        y_speed = (y_speed[1:, :-1] - y_speed[:-1, :-1])/self.dx
        self._plot_2d(y_speed - x_speed)
        pyplot.title("Vorticity")
        pyplot.show()
        
    def linearized_distribution(self, A, vx=0.0, vy=0.0, t=0.0):
        try:
            return A[:, :, numpy.newaxis]*(1.0 - self.m*(vx[:, :, numpy.newaxis]*(self.dv*self.vxhat).flatten() + vy[:, :, numpy.newaxis]*(self.dv*self.vxhat.T).flatten())/(self.k*self.t_0) + self.m*t[:, :, numpy.newaxis]*(((self.dv*self.vxhat).flatten())**2 + ((self.dv*self.vxhat.T).flatten())**2)/(2*self.k*self.t_0**2))*self._distribution_static
        except (IndexError, TypeError):
            return A*(1.0 - self.m*(vx*(self.dv*self.vxhat).flatten() + vy*(self.dv*self.vxhat.T).flatten())/(self.k*self.t_0) + self.m*t*(((self.dv*self.vxhat).flatten())**2 + ((self.dv*self.vxhat.T).flatten())**2)/(2*self.k*self.t_0**2))*self._distribution_static
    
    def mixing(self):
        M0 = numpy.sum(self.state_space, axis=2)
        M1x = numpy.sum(self.state_space*(-self.dv*self.vxhat).flatten(), axis=2)
        M1y = numpy.sum(self.state_space*(-self.dv*self.vxhat).T.flatten(), axis=2)
        M2 = numpy.sum(self.state_space*self.dv**2*(self.vxhat**2 + self.vxhat.T**2).flatten(), axis=2)
        A = self.find_A(M0, M2)
        vx = self.find_vx(M0, M1x, M2)
        vy = self.find_vy(M0, M1y, M2)
        t = self.find_t(M0, M2)
        self.state_space = self.state_space + self.mixing_rate*(self.linearized_distribution(A, vx, vy, t) - self.state_space)
    
    def diffusion(self):
        state_space = numpy.zeros(shape=(self.resolution, self.resolution, 9))
        state_space[:-2, :-2, 0] = self.state_space[1:-1, 1:-1, 0]
        state_space[1:-1, :-2, 1] = self.state_space[1:-1, 1:-1, 1]
        state_space[2:, :-2, 2] = self.state_space[1:-1, 1:-1, 2]

        state_space[:-2, 1:-1, 3] = self.state_space[1:-1, 1:-1, 3]
        state_space[1:-1, 1:-1, 4] = self.state_space[1:-1, 1:-1, 4]
        state_space[2:, 1:-1, 5] = self.state_space[1:-1, 1:-1, 5]

        state_space[:-2, 2:, 6] = self.state_space[1:-1, 1:-1, 6]
        state_space[1:-1, 2:, 7] = self.state_space[1:-1, 1:-1, 7]
        state_space[2:, 2:, 8] = self.state_space[1:-1, 1:-1, 8]

        for i in range(1, self.resolution - 1):
            state_space[0][i - 1][2] = self.state_space[0][i][0]
            state_space[0][i][5] = self.state_space[0][i][3]
            state_space[0][i + 1][8] = self.state_space[0][i][6]

            state_space[0][i - 1][1] = self.state_space[0][i][1]
            state_space[1][i - 1][2] = self.state_space[0][i][2]
            state_space[1][i][5] = self.state_space[0][i][5]
            state_space[0][i + 1][7] = self.state_space[0][i][7]
            state_space[1][i + 1][8] = self.state_space[0][i][8]
            
            state_space[0][i][4] = self.state_space[0][i][4]
            state_space[-1][i][4] = self.state_space[-1][i][4]
            
            state_space[-1][i - 1][0] = self.state_space[-1][i][2]
            state_space[-1][i][3] = self.state_space[-1][i][5]
            state_space[-1][i + 1][6] = self.state_space[-1][i][8]

            state_space[-2][i - 1][0] = self.state_space[-1][i][0]
            state_space[-1][i - 1][1] = self.state_space[-1][i][1]
            state_space[-2][i][3] = self.state_space[-1][i][3]
            state_space[-2][i + 1][6] = self.state_space[-1][i][6]
            state_space[-1][i + 1][7] = self.state_space[-1][i][7]

        for i in range(1, self.resolution - 1):
            state_space[i - 1][0][6] = self.state_space[i][0][0]
            state_space[i][0][7] = self.state_space[i][0][1]
            state_space[i + 1][0][8] = self.state_space[i][0][2]

            state_space[i - 1][0][3] = self.state_space[i][0][3]
            state_space[i + 1][0][5] = self.state_space[i][0][5]
            state_space[i - 1][1][6] = self.state_space[i][0][6]
            state_space[i][1][7] = self.state_space[i][0][7]
            state_space[i + 1][1][8] = self.state_space[i][0][8]

            state_space[i][0][4] = self.state_space[i][0][4]
            state_space[i][-1][4] = self.state_space[i][-1][4]

            state_space[i - 1][-1][0] = self.state_space[i][-1][6]
            state_space[i][-1][1] = self.state_space[i][-1][7]
            state_space[i + 1][-1][2] = self.state_space[i][-1][8]

            state_space[i - 1][-2][0] = self.state_space[i][-1][0]
            state_space[i][-2][1] = self.state_space[i][-1][1]
            state_space[i + 1][-2][2] = self.state_space[i][-1][2]
            state_space[i - 1][-1][3] = self.state_space[i][-1][3]
            state_space[i + 1][-1][5] = self.state_space[i][-1][5]

        state_space[0][0][8] = self.state_space[0][0][0]
        state_space[0][0][7] = self.state_space[0][0][1]
        state_space[1][0][8] = self.state_space[0][0][2]
        state_space[0][0][5] = self.state_space[0][0][3]
        state_space[0][1][8] = self.state_space[0][0][6]

        state_space[1][0][5] = self.state_space[0][0][5]
        state_space[0][1][7] = self.state_space[0][0][7]
        state_space[1][1][8] = self.state_space[0][0][8]
        
        state_space[0][0][4] = self.state_space[0][0][4]
        state_space[-1][-1][4] = self.state_space[-1][-1][4]

        state_space[-1][-2][0] = self.state_space[-1][-1][2]
        state_space[-1][-1][3] = self.state_space[-1][-1][5]
        state_space[-2][-1][0] = self.state_space[-1][-1][6]
        state_space[-1][-1][1] = self.state_space[-1][-1][7]
        state_space[-1][-1][0] = self.state_space[-1][-1][8]

        state_space[-2][-2][0] = self.state_space[-1][-1][0]
        state_space[-1][-2][1] = self.state_space[-1][-1][1]
        state_space[-2][-1][3] = self.state_space[-1][-1][3]



        state_space[-2][0][6] = self.state_space[-1][0][0]
        state_space[-1][0][7] = self.state_space[-1][0][1]
        state_space[-1][0][6] = self.state_space[-1][0][2]
        state_space[-1][0][3] = self.state_space[-1][0][5]
        state_space[-1][1][6] = self.state_space[-1][0][8]

        state_space[-2][0][3] = self.state_space[-1][0][3]
        state_space[-2][1][6] = self.state_space[-1][0][6]
        state_space[-1][1][7] = self.state_space[-1][0][7]
        
        state_space[-1][0][4] = self.state_space[-1][0][4]
        state_space[0][-1][4] = self.state_space[0][-1][4]

        state_space[0][-2][2] = self.state_space[0][-1][0]
        state_space[0][-1][5] = self.state_space[0][-1][3]
        state_space[0][-1][2] = self.state_space[0][-1][6]
        state_space[0][-1][1] = self.state_space[0][-1][7]
        state_space[1][-1][2] = self.state_space[0][-1][8]

        state_space[0][-2][1] = self.state_space[0][-1][1]
        state_space[1][-2][2] = self.state_space[0][-1][2]
        state_space[1][-1][5] = self.state_space[0][-1][5]
        
        self.state_space = state_space

size = 200
state = LBM(size, 0.9)

density = 200.0
t = 10.0
vx = 0.0
vy = 0.0
R2 = state.const_2space
state.state_space[:, :] = state.linearized_distribution(1.0*R2, 0.0*R2, 0.0*R2, 0.0*R2)
state.state_space[int(size/2)-45:int(size/2)-35, int(size/2)-45:int(size/2)-35] = state.linearized_distribution(density, vx, vy, t)
state.state_space[int(size/2)+35:int(size/2)+45, int(size/2)+15:int(size/2)+25] = state.linearized_distribution(density, vx, vy, t)

fig = pyplot.figure()
j = 0

def mov(*args):
    global j
    j = j + 1
    pyplot.clf()
    state.diffusion()
    state.mixing()
    print(j, "/ " + str(30*60))

    return state.plot_moment_1(vmax=6.0)

from matplotlib import animation
ani = animation.FuncAnimation(fig, mov, interval=20, frames=30*60, blit=True)
ani.save("test2_0.9.mp4", fps=30, bitrate=1600)
#pyplot.show()
