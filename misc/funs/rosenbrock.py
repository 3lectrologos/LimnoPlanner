import numpy as np

lim = {'x1': (0, 1), 'x2': (0, 1)}

def eval(x):
    f = lambda x: 10 - 100*(x[1]-x[0]**2)**2 - (1-x[0])**2
    return np.apply_along_axis(f, 1, x).reshape(-1, 1)
