import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d


class Testcase(object):
    def __init__(self, kernel, fun=None, data=None, h=0, name='untitled'):
        if not fun and not data:
            raise Exception('Need to provide either fun or data')
        if fun:
            self.type = 'fun'
            self.fun = fun
            self.lim = fun.lim
        else:
            self.type = 'data'
            self.data = data
        self.kernel = kernel
        self.name = name
        self.h = h

    def grid(self, ngrid=30):
        x1 = np.linspace(self.lim['x'][0], self.lim['x'][1], ngrid)
        x2 = np.linspace(self.lim['y'][0], self.lim['y'][1], ngrid)
        x1, x2 = np.meshgrid(x1, x2)
        x1r = x1.reshape((-1, 1));
        x2r = x2.reshape((-1, 1));
        x = np.concatenate((x1r, x2r), 1)
        y = self.fun.eval(x)
        y = y.reshape((ngrid, ngrid))
        return (x1, x2, y)

    def surf(self):
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        (x1, x2, y) = self.grid()
        surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=mpl.cm.jet,
                               linewidth=0.5, antialiased=True, shade=True)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$y$')
        plt.show()

    def plot(self):
        self.plot_aux('all')

    def ploth(self):
        self.plot_aux('this')

    def plot_aux(self, mode):
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.gca()
        (x1, x2, y) = self.grid()
        if mode == 'all':
            plt.contourf(x1, x2, y, 20, cmap=plt.cm.jet)
            plt.contour(x1, x2, y, 20, colors='k', linestyles='solid')
        elif mode == 'this':
            plt.contourf(x1, x2, y, cmap=plt.cm.jet,
                         levels=[-100000000.0, self.h, 100000000.0])
            plt.contour(x1, x2, y, colors='k', linestyles='solid',
                        levels=[self.h])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        plt.show()

def test():
    import infpy.gp.kernel_short_names as kernels
    import misc.funs.rosenbrock
    k = kernels.SE([1, 1])
    tc = Testcase(k, fun=misc.funs.rosenbrock, h=-10.0)
    return tc
    #tc.surf()
    #tc.plot()
    #tc.ploth()
