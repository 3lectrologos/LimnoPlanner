import numpy as np
import gp.gp as gp
import gp.kernels as kernels
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d


class Testcase(object):
    def __init__(self, kernel, fun=None, data=None, h=0, name='untitled'):
        if not fun and not data:
            raise Exception('Need to provide either fun or data')
        self.kernel = kernel
        if fun:
            self.type = 'fun'
            self.fun = fun
            self.lim = fun.lim
            xtrain = np.zeros((0, 2))
            ytrain = np.zeros((0, 1))
        else:
            self.type = 'data'
            self.data = data
            self.lim = {}
            self.lim['x1'] = (min(self.data['x'][:,0]),
                              max(self.data['x'][:,0]))
            self.lim['x2'] = (min(self.data['x'][:,1]),
                              max(self.data['x'][:,1]))
            xtrain = self.data['x']
            ytrain = self.data['y']
        self.model = gp.GP(kernel)
        self.model.add(xtrain, ytrain)
        self.name = name
        self.h = h

    @classmethod
    def from_mat(cls, fname):
        import scipy.io
        m = scipy.io.loadmat(fname, byte_order='native')
        data = {}
        data['x'] = m['tc'][0][0][0][0][0][0].newbyteorder('=')
        data['y'] = m['tc'][0][0][0][0][0][1].newbyteorder('=')
        h = m['tc'][0][0][1][0][0].newbyteorder('=')
        name = m['tc'][0][0][3][0]
        hyp = {'mean': 2.4, 'cov': [6, 1, 0.7], 'lik': -0.7}
        k = kernels.SE(hyp)
        tc = Testcase(k, data=data, h=h, name=name)
        return tc

    def grid(self, ngrid=30):
        x1 = np.linspace(self.lim['x1'][0], self.lim['x1'][1], ngrid)
        x2 = np.linspace(self.lim['x2'][0], self.lim['x2'][1], ngrid)
        x1, x2 = np.meshgrid(x1, x2)
        x1r = x1.reshape((-1, 1));
        x2r = x2.reshape((-1, 1));
        x = np.concatenate((x1r, x2r), 1)
        if self.type == 'fun':
            y = self.fun.eval(x)
        else:
            (y, _) = self.model.inf(x)
            y = np.array(y)
        y = y.reshape((ngrid, ngrid))
        return (x1, x2, y)

    def surf(self, ngrid=30):
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        (x1, x2, y) = self.grid(ngrid)
        surf = ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=mpl.cm.jet,
                               linewidth=0.5, antialiased=True, shade=True)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$y$')
        plt.show()

    def plot(self, ngrid=30):
        self.plot_aux('all', ngrid)

    def ploth(self, ngrid=30):
        self.plot_aux('this', ngrid)

    def plot_aux(self, mode, ngrid):
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.gca()
        (x1, x2, y) = self.grid(ngrid)
        if mode == 'all':
            plt.contourf(x1, x2, y, 20, cmap=plt.cm.jet)
            plt.contour(x1, x2, y, 20, colors='k', linestyles='solid')
        elif mode == 'this':
            plt.contourf(x1, x2, y, cmap=plt.cm.jet,
                         levels=[-1000000.0, self.h, 1000000.0])
            plt.contour(x1, x2, y, colors='k', linestyles='solid',
                        levels=[self.h])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        plt.show()

def test():
    import misc.funs.rosenbrock
    hyp = {'mean': 0, 'cov': [-1.5, -1.5, 5], 'lik': -1}
    k = kernels.SE(hyp)
    tc = Testcase(k, fun=misc.funs.rosenbrock, h=-10.0)
    #tc.surf()
    #tc.plot()
    tc.ploth()
    return tc

def test2():
    fn = '/home/alkis/gp/testcases/tc/tc_limnolog-00110714-093200_bgape_gp.mat'
    tc = Testcase.from_mat(fn)
    tc.surf(50)
    return tc
