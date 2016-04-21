import numpy
import scipy.ndimage
from matplotlib import pyplot

class vec:
    def __init__(self, x, y=None):
        if y == None:
            if len(x) == 2:
                self.coord = numpy.array(x)
        else:
            self.coord = numpy.array([x, y])
    
    def perp(self):
        return vec(-self.coord[1], self.coord[0])

    def dot(self, other):
        return numpy.sum(self.coord*other.coord)
    
    def project_on(self, other):
        return (self.dot(other)/other.dot(other))*other
    
    def length(self):
        return numpy.sqrt(self._length2())
    
    def _length2(self):
        return self.dot(self)
    
    def __eq__(self, other):
        return all(self.coord == other.coord)
    
    def __ne__(self, other):
        return any(self.coord != other.coord)
    
    def __lt__(self, other):
        return self._length2() < other._length2()        

    def __gt__(self, other):
        return self._length2() > other._length2()        

    def __le__(self, other):
        return self._length2() <= other._length2()        

    def __ge__(self, other):
        return self._length2() >= other._length2()
    
    # math operations
    def __add__(self, other):
        return vec(self.coord + other.coord)

    def __radd__(self, other):
        return vec(self.coord + other.coord)

    def __sub__(self, other):
        return vec(self.coord - other.coord)

    def __rsub__(self, other):
        return vec(self.coord - other.coord)

    def __mul__(self, factor):
        return vec(self.coord*factor)
        
    def __rmul__(self, factor):
        return vec(self.coord*factor)
        
    def __truediv__(self, factor):
        return vec(self.coord/factor)

    def plot(self, axis=None):
        if axis == None:
            axis = pyplot
        axis.scatter([self.coord[0]], [self.coord[1]])
        
class ray:
    def __init__(self, origin, unitvec):
        self.origin = origin
        self.unitvec = unitvec/unitvec.length()
    
    def to_line(self, length):
        return line(self.origin, length*self.unitvec + self.origin)
    
    def intersect(self, other):
        diff = other.origin - self.origin
        y_0 = diff.dot(self.unitvec.perp())
        dy = other.unitvec.dot(self.unitvec.perp())
        t = -y_0/dy        
        return other.origin + t*other.unitvec
    
    def plot(self, axis=None):
        if axis == None:
            axis = pyplot
        axis.scatter([self.origin.coord[0]], [self.origin.coord[1]])
        axis.plot([self.origin.coord[0], self.origin.coord[0] + self.unitvec.coord[0]], [self.origin.coord[1], self.origin.coord[1] + self.unitvec.coord[1]], linestyle="--")    
        
class line:
    def __init__(self, A, B):
        if (A > B):
            A, B = B, A
        self.A = A
        self.B = B
    
    def midvec(self):
        return vec(0.5*(self.A.coord + self.B.coord))
    
    def bisector(self):
        direction = self.B.coord - self.A.coord
        return ray(self.midvec(), vec(direction[1], -direction[0]))
    
    def to_ray(self):
        return ray(self.A, self.B - self.A)

    def __eq__(self, other):
        return (self.A == other.A) and (self.B == other.B)
    
    def __ne__(self, other):
        return (self.A != other.A) or (self.B != other.B)
        
    def plot(self, axis=None):
        if axis == None:
            axis = pyplot
        self.A.plot(axis)
        self.B.plot(axis)
        axis.plot([self.A.coord[0], self.B.coord[0]], [self.A.coord[1], self.B.coord[1]], linestyle="-")            

class triangle:
    def __init__(self, A, B, C):
        # reorder vecs
        if (A > B):
            A, B = B, A
        if (B > C):
            B, C = C, B
        if (A > B):
            A, B = B, A
        self.vertice = [A, B, C]
        self.lines = [line(A, B), line(B, C), line(C, A)]
        self._circumcircle()

    def _circumcircle(self):
        ray1 = self.lines[0].bisector()
        ray2 = self.lines[1].bisector()
        self.center = ray1.intersect(ray2)
        self.radius = (self.vertice[0] - self.center).length()
    
    def __eq__(self, other):
        return all(self.vertice[i] == other.vertice[i] for i in range(len(self.vertice)))

    def __ne__(self, other):
        return any(self.vertice[i] != other.vertice[i] for i in range(len(self.vertice)))

    def plot(self, axis=None):
        if axis == None:
            axis = pyplot
        for line in self.lines:
            line.plot(axis)
        circle = pyplot.Circle(self.center.coord, self.radius, fill=False)
        
#        if axis == pyplot:
#            axis.gca().add_artist(circle)
#        else:
#            axis.add_artist(circle)

class bowyer_watson:
    def __init__(self, size):
        self.size = size
        A = vec(-numpy.sqrt(6)*self.size/2, -self.size/numpy.sqrt(2))
        B = vec(numpy.sqrt(6)*self.size/2, -self.size/numpy.sqrt(2))
        C = vec(0, 2*self.size/numpy.sqrt(2))
        self.border = [line(A, B), line(B, C), line(C, A)]
        self.triangles = [triangle(A, B, C)]
    
    def add_point(self, point):
        if (abs(point.coord[0]) > self.size/2) or (abs(point.coord[1]) > self.size/2):
            raise ValueError("point: " + str(point) + " out size of bounds for square with length: " + str(self.size) + " centered at the origin.")
        
        bad_triangles = [triangle for triangle in self.triangles if ((point - triangle.center)._length2() < triangle.radius**2)]
        self.triangles = [triangle for triangle in self.triangles if not (triangle in bad_triangles)]

        polygon=[]
        for bad_triangle in bad_triangles:
            for line in bad_triangle.lines:
                line_in_polygon = False
                if line in polygon:
                    i = polygon.index(line)
                    del polygon[i]
                else:
                    polygon = polygon + [line]
            
        for line in polygon:
            self.triangles = self.triangles + [triangle(line.A, line.B, point)]
    
    def get_voronoi(self, resolution = 100):
        triangles = self.triangles

        outside=[]
        triangle_outside = []
        triangle_inside = []
        for triangle in triangles:
            for tline in triangle.lines:
                line_in_outside = False
                if tline in outside:
                    i = outside.index(tline)
                    triangle_inside = triangle_inside + [(triangle, triangle_outside[i])]
                    del outside[i]
                    del triangle_outside[i]
                else:
                    outside = outside + [tline]
                    triangle_outside = triangle_outside + [triangle]
        lines = []
        for triangle_pair in triangle_inside:
            A, B = triangle_pair[0].center, triangle_pair[1].center
            lines = lines + [line(A, B)]
        return voronoi(lines, [-self.size/2, self.size/2], resolution)
    
    def plot(self, axis=None, setup_only=False):
        if axis == None:
            axis = pyplot
        pyplot.plot([-self.size/2, -self.size/2, self.size/2, self.size/2, -self.size/2], [-self.size/2, self.size/2, self.size/2, -self.size/2, -self.size/2], linestyle="--")
        circle = pyplot.Circle((0., 0.), numpy.sqrt(2)*self.size/2, fill=False)
        
        if axis == pyplot:
            axis.gca().add_artist(circle)
        else:
            axis.add_artist(circle)
        
        for line in self.border:
            line.plot(axis)
            
        if not setup_only:
            for triangle in self.triangles:
                triangle.plot(axis)

class voronoi:
    def __init__(self, lines, limit=[-1, 1], resolution=100):
        self.lines = lines
        self.limit = limit
        self.resolution = resolution
        self.raster = None
        self.regions = None
        self.neighbors = {}
        self.max_id = 0
        self.filter_max = 0.2

    def rasterize(self):
        self.raster = numpy.zeros((self.resolution, self.resolution))
        self.regions = numpy.zeros((self.resolution, self.resolution), dtype=int)
        self.neighbors = {}
        self.max_id = 0

        for line in self.lines:
            self.bresenham_line(self.map_line(line))
            
    def find_regions(self):
        region_id = 1
        boundry = [(0, 0)]
        while len(boundry) != 0:
            x, y = boundry[0]
            boundry = boundry[1:]
            if self.canvas(x, y + 1):
                if self.regions[x, y + 1] == 0:
                    boundry, region_id = self.find_region(region_id, x, y + 1, boundry)
            if self.canvas(x + 1, y):
                if self.regions[x + 1, y] == 0:
                    boundry, region_id = self.find_region(region_id, x + 1, y, boundry)
        
    def find_region(self, region_id, x_0, y_0, boundry=[]):
        if self.raster[x_0, y_0]:
            return [(x_0, y_0)] + boundry, region_id
        
        temp = [(x_0, y_0)]
        while len(temp) != 0:
            x, y = temp[0]
            temp = temp[1:]
            self.regions[x, y] = region_id
            if self.is_clear(x, y + 1) and (not (x, y + 1) in temp):
                temp += [(x, y + 1)]
            elif not self.is_clear(x, y + 1, False):
                if not (x, y + 1) in boundry:
                    boundry += [(x, y + 1)]
                if self.canvas(x, y + 2):
                    other_id = self.regions[x, y + 2]
                    self._add_neighbor(region_id, other_id)
                
            if self.is_clear(x, y - 1) and (not (x, y - 1) in temp):
                temp += [(x, y - 1)]
            elif not self.is_clear(x, y - 1, False):
                if not (x, y - 1) in boundry:
                    boundry += [(x, y - 1)]
                if self.canvas(x, y - 2):
                    other_id = self.regions[x, y - 2]
                    self._add_neighbor(region_id, other_id)
            
            if self.is_clear(x + 1, y) and (not (x + 1, y) in temp):
                temp += [(x + 1, y)]
            elif not self.is_clear(x + 1, y, False):
                if not (x + 1, y) in boundry:
                    boundry += [(x + 1, y)]
                if self.canvas(x + 2, y):
                    other_id = self.regions[x + 2, y]
                    self._add_neighbor(region_id, other_id)
                
            if self.is_clear(x - 1, y) and (not (x - 1, y) in temp):
                temp += [(x - 1, y)]
            elif not self.is_clear(x - 1, y, False):
                if not (x - 1, y) in boundry:
                    boundry += [(x - 1, y)]
                if self.canvas(x - 2, y):
                    other_id = self.regions[x - 2, y]
                    self._add_neighbor(region_id, other_id)
        
        self.max_id = region_id
        return boundry, region_id + 1
        
    def mean_distance(self, resolution=None):
        if resolution != None:
            self.resolution = resolution
        self.rasterize()
        self.find_regions()
    
        centroids = []
        for i in range(1, self.max_id + 1):
            temp = numpy.zeros((self.resolution, self.resolution), dtype=int)
            temp[self.regions == i] = 1
            total = numpy.sum(temp)
            y = numpy.sum(temp*numpy.linspace(0, self.resolution - 1, self.resolution)[:, numpy.newaxis])/total
            x = numpy.sum(temp*numpy.linspace(self.resolution - 1, 0, self.resolution)[numpy.newaxis, :])/total
            centroids += [(y, x)]

        evaluated = []
        distances = []
        for i in self.neighbors.keys():
#        for i in range(1, self.max_id + 1):
            for j in self.neighbors[i]:
                a, b = i - 1, j - 1
                if a > b:
                    a, b = b, a
                if not (a, b) in evaluated:
                    evaluated += [(a, b)]
                    dx = centroids[a][0] - centroids[b][0]
                    dy = centroids[a][1] - centroids[b][1]
                    distances += [numpy.sqrt(dx**2 + dy**2)]
        distances = numpy.array(distances)
        return self.inv_map(numpy.mean(distances)), self.inv_map(numpy.std(distances)/numpy.sqrt(len(distances)))
        
    def bresenham_line(self, line):
        dx = numpy.abs(line.A.coord[0] - line.B.coord[0])
        dy = numpy.abs(line.A.coord[1] - line.B.coord[1])
        x, y = line.A.coord[0], line.A.coord[1]
        sx = -1 if line.A.coord[0] > line.B.coord[0] else 1
        sy = -1 if line.A.coord[1] > line.B.coord[1] else 1

        if dx > dy:
            err = dx/2.
            while int(numpy.floor(x)) != int(numpy.floor(line.B.coord[0])):
                if self.canvas(numpy.floor(x), numpy.floor(y)):
                    self.raster[int(numpy.floor(x)), int(numpy.floor(y))] = 1.0
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy/2.
            while int(numpy.floor(y)) != int(numpy.floor(line.B.coord[1])):
                if self.canvas(numpy.floor(x), numpy.floor(y)):
                    self.raster[int(numpy.floor(x)), int(numpy.floor(y))] = 1.0
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        if self.canvas(numpy.floor(x), numpy.floor(y)):
            self.raster[int(numpy.floor(x)), int(numpy.floor(y))] = 1.0
    
    def map_point(self, point):
        return vec(self.resolution*(point.coord - numpy.array([self.limit[0], self.limit[0]]))/numpy.array([self.limit[1] - self.limit[0], self.limit[1] - self.limit[0]]))
    
    def inv_map(self, x):
        return x*(self.limit[1] - self.limit[0])/self.resolution        

    def map_line(self, unmapped):
        return line(self.map_point(unmapped.A), self.map_point(unmapped.B))
    
    def canvas(self, x, y):
        return (0 <= x < self.resolution) and (0 <= y < self.resolution)
        
    def _add_neighbor(self, region_id, other_id):
        if (region_id != 0) and (other_id != 0):
            if region_id in self.neighbors:
                if not other_id in self.neighbors[region_id]:
                    self.neighbors[region_id] += [other_id]
            else:
                self.neighbors[region_id] = [other_id]
    
            if other_id in self.neighbors:
                if not region_id in self.neighbors[other_id]:
                    self.neighbors[other_id] += [region_id]
            else:
                self.neighbors[other_id] = [region_id]

    def is_clear(self, x, y, strict=True):
        if strict:
            if self.canvas(x, y):
                if (self.raster[x, y] == 0) and (self.regions[x, y] == 0):
                    return True
            return False
        else:        
            if self.canvas(x, y):
                return self.raster[x, y] == 0
            return True
            
    def gaussian(self, width, bounds, resolution=None):
        if resolution != None:
            self.resolution = resolution
        self.rasterize()

        efall_px = (width/2.)*self.resolution/(self.limit[1] - self.limit[0])
        range_px = int((bounds/2.)*self.resolution/(self.limit[1] - self.limit[0]))
        if range_px < 1:
            raise ValueError("range is too small, filter must be larger than 2x2")
        if efall_px > range_px:
            raise ValueError("gausian width larger than filter bounds")
        if range_px > self.resolution*self.filter_max/2.:
            raise ValueError("filter bounds larger than " + str(self.filter_max) + " * resolution.")
        
        X, Y = numpy.meshgrid(numpy.linspace(-range_px, range_px, 2*range_px + 1), numpy.linspace(-range_px, range_px, 2*range_px + 1))
        filter = numpy.exp(-(X**2 + Y**2)/(2.0*efall_px**2))
        filter = filter/numpy.sum(filter)
        self.raster = scipy.ndimage.filters.gaussian_filter(self.raster, efall_px, truncate=range_px/efall_px)
        self.raster[self.raster > numpy.sum(filter[int(len(filter)/2)])] = numpy.sum(filter[int(len(filter)/2)])
        self.raster = self.raster/numpy.max(self.raster)

    def plot_raster(self, axis=None):
        if axis == None:
            axis = pyplot
        axis.imshow(numpy.transpose(self.raster, axes=(1, 0))[::-1], interpolation="none")
        
    def plot_region(self, axis=None):
        if axis == None:
            axis = pyplot
        axis.imshow(numpy.transpose(self.regions, axes=(1, 0))[::-1], interpolation="none")
        
    def plot_vector(self, axis=None):
        if axis == None:
            axis = pyplot
        for line in self.lines:
            line.plot(axis)

""" 
import random
random.seed(0)
r = 2.0
bw=bowyer_watson(r)
for _ in range(10):
    bw.add_point(vec(random.uniform(-r/2, r/2), random.uniform(-r/2, r/2)))

#bw.plot()
#pyplot.show()
v = bw.get_voronoi(2000)
#bw.plot(setup_only = True)
#v.plot_vector()
#pyplot.show()
v.gaussian(0.04, 0.10)
#v.plot_raster()
#pyplot.show()
print(v.mean_distance(400))
"""
