from matplotlib import pyplot
import numpy

class LBM:
    def __init__(self, resolution, mixing_rate=0.1):
        self.resolution = resolution
        self.mixing_rate = mixing_rate
        self.interpolation="none"
        self.colorbar = pyplot.cm.gray
        self.conserve_first_moment = True

        self.dx = 1.0 # in m
        self.m = 1.0 # in kg
        self.dv = 1.0 # in m/s
        self.t_0 = 200.0 # in K
#        self.t_0 = 2000.0 # in K
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
        
        self.direction_slice = {"0":slice(4, 5), 
                                "y":slice(1, 8, 6), 
                                "y-":slice(0, 3), 
                                "y+":slice(6, 9), 
                                "x":slice(3, 6, 2), 
                                "x+":slice(2, 9, 3), 
                                "x-":slice(0, 7, 3)} 
        
        self.direction_shift = {"0":(slice(None, None), slice(None, None)), 
                                "y+":(slice(None, None), slice(1, None)), 
                                "y-":(slice(None, None), slice(None, -1)), 
                                "x+":(slice(1, None), slice(None, None)), 
                                "x-":(slice(None, -1), slice(None, None))}
        
        self._distribution_static = numpy.array([self.e_0**2, self.e_0, self.e_0**2, self.e_0, 1.0, self.e_0, self.e_0**2, self.e_0, self.e_0**2])
        self.state_space = numpy.zeros(shape=(self.resolution, self.resolution, 9))
        self.blank_state = numpy.ones(shape=(self.resolution, self.resolution, 9))
        self.space_ones = numpy.ones(shape=(self.resolution, self.resolution))
        
        # boundry conditions: by if no boundry condition is defined
        # a periodic is assumed. A such points with no defined boundry
        # condition must be placed symmetricaly
        self.boundry = numpy.zeros(shape=(self.resolution + 2, self.resolution + 2), dtype=bool)
        self.boundry_periodic = numpy.zeros(shape=(self.resolution + 2, self.resolution + 2), dtype=bool)
        self.boundry_slip = numpy.zeros(shape=(self.resolution + 2, self.resolution + 2), dtype=bool)
        self.boundry_stp = numpy.zeros(shape=(self.resolution + 2, self.resolution + 2), dtype=bool)
        
        self.vxhat = numpy.linspace(-1.0, 1.0, 3)*numpy.ones(shape=(3))[:, numpy.newaxis]
        
        if (mixing_rate < 0.0) or (mixing_rate > 1.0):
            raise ValueError("Invalid mixing rate (" + str(mixing_rate) + "). Mixing must be within [0.0, 1.0].")
        
        self.stp_state_space = self.linearized_distribution(1.0, 0.0, 0.0, 0.0)
        self.initalize()
    
    # methods to compute spectrum distributions
    def find_A(self, Moment0, Moment2):
        return self.a_0*Moment2 + self.a_1*Moment0
    
    def find_vx(self, Moment0, Moment1x, Moment2):
        return self.b_0*Moment1x/(Moment2*self.b_1 + self.b_2*Moment0)
    
    def find_vy(self, Moment0, Moment1y, Moment2):
        return self.b_0*Moment1y/(Moment2*self.b_1 + self.b_2*Moment0)
    
    def find_t(self, Moment0, Moment2):
        return self.c_0*(self.c_1*Moment2 + self.c_2*Moment0)/(self.c_3*Moment2 + self.c_4*Moment0)

    def set_stp_distribution(self, density=1.0, vx=0.0, vy=0.0, t=0.0):
        self.stp_state_space = self.linearized_distribution(density, vx, vy, t)
    
    #plotting functions
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

    def plot_moment_1(self, norm=False):
        if norm:
            self._plot_2d(numpy.sqrt(numpy.sum(self.state_space*(-self.dv*self.vxhat).flatten(), axis=2)**2 + numpy.sum(self.state_space*(-self.dv*self.vxhat.T).flatten(), axis=2)**2)/numpy.sum(self.state_space, axis=2))
        else:
            self._plot_2d(numpy.sqrt(numpy.sum(self.state_space*(-self.dv*self.vxhat).flatten(), axis=2)**2 + numpy.sum(self.state_space*(-self.dv*self.vxhat.T).flatten(), axis=2)**2))
        pyplot.title("First Moment")
        pyplot.show()
        
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
    
    # boundry initalization and handeling
    def initalize(self):
        default_border = numpy.ones(shape=(self.resolution + 2, self.resolution + 2), dtype=bool)
        default_border[1:-1, 1:-1] = False
        self.boundry_stp = numpy.logical_and(self.boundry_stp, numpy.logical_not(self.boundry_slip))
        self.boundry_periodic = numpy.logical_and(default_border, numpy.logical_not(numpy.logical_or(self.boundry_slip, self.boundry_stp)))
        self.boundry = numpy.logical_or(self.boundry_periodic, numpy.logical_or(self.boundry_slip, self.boundry_stp))
        self.state_space = self.state_space*numpy.logical_not(self.boundry[1:-1, 1:-1])[:, :, numpy.newaxis]
        self.state_space[:, :, 4] += 1.0E-8*self.boundry[1:-1, 1:-1]
        if (self.boundry_periodic[:, 0] != self.boundry_periodic[:, -1]).any() or (self.boundry_periodic[0, :] != self.boundry_periodic[-1, :]).any():
            raise ValueError("Periodic boundry condition is non symetric.")
    
    # boundry evaluation
    def _direction_flip(self, direction):
        if direction == "x+":
            return "x-"
        elif direction == "x-":
            return "x+"
        elif direction == "y+":
            return "y-"
        elif direction == "y-":
            return "y+"
        else:
            return direction
    
    def _shift_boundry(self, boundry, direction):
        if direction == "0":
            boundry = boundry[1:-1, 1:-1]
        elif direction == "x+":
            boundry = boundry[2:, 1:-1]
        elif direction == "x-":
            boundry = boundry[:-2, 1:-1]
        elif direction == "y+":
            boundry = boundry[1:-1, 2:]
        elif direction == "y-":
            boundry = boundry[1:-1, :-2]
        else:
            raise ValueError("direction identifier \'" + direction + "\' is invalid.") 
        return boundry
    
    def _shift_elements(self, field, direction):
        masked_state_space = numpy.zeros(shape=(self.resolution, self.resolution, 9))
        masked_state_space[self.direction_shift[direction] + (slice(None, None), )] = (field*numpy.logical_not(self._shift_boundry(self.boundry, direction))[:, :, numpy.newaxis])[self.direction_shift[self._direction_flip(direction)] + (slice(None, None), )]
        masked_state_space[self.boundry[1:-1, 1:-1], 4] = 1.0E-8
        return masked_state_space
    
    def _eval_periodic_boundry(self, field, direction):
        masked_state_space = field*self._shift_boundry(self.boundry_periodic, direction)[:, :, numpy.newaxis]
        if "x" in direction:
            flip = (slice(None, None, -1), slice(None, None))
        elif "y" in direction:
            flip = (slice(None, None), slice(None, None, -1))
        return masked_state_space[flip]

    def _eval_stp_boundry(self, field, direction):
        masked_state_space = self.stp_state_space*self._shift_boundry(self.boundry_stp, direction)[:, :, numpy.newaxis]
        return masked_state_space

    def _eval_slip_boundry(self, field, direction):
        masked_state_space = field*self._shift_boundry(self.boundry_slip, direction)[:, :, numpy.newaxis]
        if "x" in direction:
            flip = (slice(None, None), slice(None, None), [2, 1, 0, 5, 4, 3, 8, 7, 6])
        elif "y" in direction:
            flip = (slice(None, None), slice(None, None), [6, 7, 8, 3, 4, 5, 0, 1, 2])
        return masked_state_space[flip]
    
    def diffusion(self):
        state_space = numpy.zeros(shape=(self.resolution, self.resolution, 9))
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["0"], )] = self._shift_elements(self.state_space, "0")[(slice(None, None), slice(None, None)) + (self.direction_slice["0"], )]
        state_space_tmp = numpy.zeros(shape=(self.resolution, self.resolution, 9))

        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )] = self._shift_elements(self.state_space, "x+")[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )] = self._shift_elements(self.state_space, "x-")[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )]

        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )] += self._eval_periodic_boundry(self.state_space, "x+")[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )] += self._eval_periodic_boundry(self.state_space, "x-")[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )] += self._eval_slip_boundry(self.state_space, "x+")[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )] += self._eval_slip_boundry(self.state_space, "x-")[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )] += self._eval_stp_boundry(self.state_space, "x+")[(slice(None, None), slice(None, None)) + (self.direction_slice["x-"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )] += self._eval_stp_boundry(self.state_space, "x-")[(slice(None, None), slice(None, None)) + (self.direction_slice["x+"], )]

        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] = self._shift_elements(state_space_tmp, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] = self._shift_elements(state_space_tmp, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]

        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] += self._eval_periodic_boundry(state_space_tmp, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] += self._eval_periodic_boundry(state_space_tmp, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] += self._eval_slip_boundry(state_space_tmp, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] += self._eval_slip_boundry(state_space_tmp, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] += self._eval_stp_boundry(state_space_tmp, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] += self._eval_stp_boundry(state_space_tmp, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]

        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] = self._shift_elements(self.state_space, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] = self._shift_elements(self.state_space, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]

        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] += self._eval_periodic_boundry(self.state_space, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] += self._eval_periodic_boundry(self.state_space, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] += self._eval_slip_boundry(self.state_space, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] += self._eval_slip_boundry(self.state_space, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )] += self._eval_stp_boundry(self.state_space, "y+")[(slice(None, None), slice(None, None)) + (self.direction_slice["y-"], )]
        state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )] += self._eval_stp_boundry(self.state_space, "y-")[(slice(None, None), slice(None, None)) + (self.direction_slice["y+"], )]

        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["x"], )] = state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["x"], )]
        state_space[(slice(None, None), slice(None, None)) + (self.direction_slice["y"], )] = state_space_tmp[(slice(None, None), slice(None, None)) + (self.direction_slice["y"], )]
        self.state_space = state_space
    
    # distribution dynamics and evaluations
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
        vy = self.find_vy(M0, M1y, M2) - 0.001
        t = self.find_t(M0, M2)
        self.state_space = self.state_space + self.mixing_rate*(self.linearized_distribution(A, vx, vy, t) - self.state_space)
    
size = 100
state = LBM(size, 1.0)
state.boundry_stp[:, 0] = True
state.boundry_slip[:, -1] = True

density = 200.0
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

state._plot_2d(state.boundry)
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
    for _ in range(20000):
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
