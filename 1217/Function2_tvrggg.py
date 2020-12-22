import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class function1():
    def __init__(self):
        """
        self.boundaries is the range of x and y axis
        """
        self.boundaries = np.array([-5.12, 5.12])

    def f(self, x, y):
        """
        Function (Z) value
        """
        t1 = 10 * 2
        t2 = x**2+y**2
        t3 = - 10 * np.cos(2 * np.pi * x)+10 * np.cos(2 * np.pi * y)
        return t1 + t2 + t3

class function2():
    def __init__(self):
        """
        self.boundaries is the range of x and y axis
        """
        self.boundaries = np.array([-5, 5])

    def f(self, x, y):
        """
        Function (Z) value
        """
        t1 = 100 * (y - x**2) ** 2
        t2 = (x - 1) ** 2
        return t1+t2

#function2
f2 = function2()

x = np.linspace(f2.boundaries[0], f1.boundaries[1], 4096)
y = np.linspace(f2.boundaries[0], f1.boundaries[1], 4096)
x, y = np.meshgrid(x, y)
z = f2.f(x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

fig.savefig("function2.png")