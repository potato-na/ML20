import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

class PlotFunc():
    def __init__(self, func_class):
        self.boundaries = func_class.boundaries
        self.f = func_class.f

    def plot(self):
        x = np.arange(self.boundaries[0], self.boundaries[1], 0.1)
        y = np.arange(self.boundaries[0], self.boundaries[1], 0.1)
        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        #ax.plot_wireframe(X, Y, Z)
        ax.plot_surface(X, Y, Z)
        return fig

    def show(self):
        self.plot()
        plt.show()

    def save(self, imagename):
        fig = self.plot()
        fig.savefig(imagename)

if __name__ == '__main__':
    plot_func1 = PlotFunc(function1())
    plot_func2 = PlotFunc(function2())

    plot_func1.save("Figure1.png")
    plot_func2.save("Figure2.png")

