import numpy as np

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
