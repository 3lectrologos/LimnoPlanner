import random
import numpy.matlib as np
import gp.core as gp
import gp.kernels as kernels
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d
import networkx as nx


# Constants
_GRAPH_NODE_SIZE = 30
_GRAPH_NODE_COLOR = '#434D54'
_GRAPH_XRES = 15
_GRAPH_YRES_BASIC = 7
_GRAPH_YRES_MLP = 100 / _GRAPH_YRES_BASIC
_GRAPH_YRES_EXTRA = 7
_GRAPH_ASPECT_RATIO = 7

def test():
    import misc.funs.rosenbrock
    hyp = {'mean': 0, 'cov': [-1.5, -1.5, 5], 'lik': -1}
    k = kernels.SE(hyp)
    tc = Testcase(k, fun=misc.funs.rosenbrock, h=-10.0)
    return tc

def test2():
    fn = '/home/alkis/gp/testcases/tc/tc_limnolog-00110714-135138_bgape_gp.mat'
    tc = Testcase.from_mat(fn)
    return tc

def graph():
    g = Graph.from_testcase(test2())
    return g

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
            self.lim['x1'] = (np.amin(self.data['x'][:,0]),
                              np.amax(self.data['x'][:,0]))
            self.lim['x2'] = (np.amin(self.data['x'][:,1]),
                              np.amax(self.data['x'][:,1]))
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
        data['x'] = np.asmatrix(m['tc'][0][0][0][0][0][0].newbyteorder('='))
        data['y'] = np.asmatrix(m['tc'][0][0][0][0][0][1].newbyteorder('='))
        h = m['tc'][0][0][1][0][0].newbyteorder('=')
        name = m['tc'][0][0][3][0]
        hyp = {'mean': 2.4, 'cov': [5.5, 1, 0.75], 'lik': -1}
        k = kernels.SE(hyp)
        tc = Testcase(k, data=data, h=h, name=name)
        return tc

    def sample(self, x):
        if self.type == 'fun':
            y = self.fun.eval(x)
        else:
            y = self.model.sample(x)
        return y

    def grid(self, ngrid=30):
        x1 = np.linspace(self.lim['x1'][0], self.lim['x1'][1], ngrid)
        x2 = np.linspace(self.lim['x2'][0], self.lim['x2'][1], ngrid)
        (x1, x2) = np.meshgrid(x1, x2)
        x1r = x1.reshape((-1, 1));
        x2r = x2.reshape((-1, 1));
        x = np.concatenate((x1r, x2r), 1)
        if self.type == 'fun':
            y = self.fun.eval(x)
        else:
            y = self.model.sample(x)
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
            plt.contourf(np.asarray(x1), np.asarray(x2), np.asarray(y),
                         20, cmap=plt.cm.jet)
            plt.contour(np.asarray(x1), np.asarray(x2), np.asarray(y),
                        20, colors='k', linestyles='solid')
        elif mode == 'this':
            plt.contourf(np.asarray(x1), np.asarray(x2), np.asarray(y),
                         cmap=plt.cm.jet, levels=[-1e10, self.h, 1e10])
            plt.contour(np.asarray(x1), np.asarray(x2), np.asarray(y),
                        colors='k', linestyles='solid', levels=[self.h])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        plt.show()

class Graph(nx.DiGraph):
    def __init__(self, x1, x2, fclass=None):
        super(Graph, self).__init__()
        self.yres = x1.shape[1]
        self.fclass = fclass
        n = 2*x1.size - 2*self.yres
        self.add_nodes_from(range(1, n+1))
        x1 = np.vstack((x1.reshape((-1, 1)), x1[-2:0:-1].reshape((-1, 1))))
        x2 = np.vstack((x2.reshape((-1, 1)), x2[-2:0:-1].reshape((-1, 1))))
        self.x = np.hstack((x1, x2))
        self.pos = dict(zip(range(1, n+1),
                            zip(x1.T.tolist()[0], x2.T.tolist()[0])))
        self.full = nx.DiGraph(self)
        for i in range(1, n+1):
            for j in self.column(i):
                dx1 = np.absolute(self.pos[j][0] - self.pos[i][0])
                dx2 = np.absolute(self.pos[j][1] - self.pos[i][1])
                if dx2 <= dx1 / _GRAPH_ASPECT_RATIO:
                    self.full.add_edge(i, j)
        tmp = 1
        self.basic = set()
        step = self.yres/_GRAPH_YRES_BASIC
        while tmp < n+1:
            self.basic = self.basic | set(self.column(tmp)[0::step])
            tmp = tmp + self.yres
        self.update_active()

    def update_active(self, inc=[]):
        self.active = self.basic | set(inc)
        if self.fclass:
            unclass = set(self.fclass(self.x)) - self.active
            tmp = 1
            step = self.yres/_GRAPH_YRES_BASIC
            while tmp <= self.number_of_nodes():
                cand = set(self.column(tmp)) & unclass
                chosen = random.sample(cand, min(_GRAPH_YRES_EXTRA, len(cand)))
                self.active = self.active | set(chosen)
                tmp = tmp + self.yres
        self.remove_edges_from(self.edges())
        for u, v in self.full.edges():
            if u in self.active and v in self.active:
                self.add_edge(u, v)

    def init_node(self):
        return 1
    
    def pcolumn(self, v, n=1):
        return [u for u in self.column(v, n) if self.has_edge(v, u)]
                    
    def column(self, v, n=1):
        assert v <= len(self)
        last = v + self.yres - 1 - (v-1) % self.yres
        fst = last + (n-1)*self.yres + 1
        nextcol = range(fst, fst+self.yres)
        return [1 + (nxt - 1) % len(self) for nxt in nextcol]

    def is_terminal(self, v):
        return v in self.column(1, 0) or v in self.column(1, _GRAPH_XRES-1)

    def rand_path(self, v, n=5):
        node = v
        path = [v]
        for i in range(n):
            node = random.choice(self.column(node))
            path.append(node)
        return path

    @classmethod
    def from_testcase(cls, tc, fclass=None):
        yres = _GRAPH_YRES_MLP*_GRAPH_YRES_BASIC + 1
        x = np.linspace(tc.lim['x1'][0], tc.lim['x1'][1], _GRAPH_XRES)
        y = np.linspace(tc.lim['x2'][0], tc.lim['x2'][1], yres)
        (x, y) = np.meshgrid(x, y)
        return Graph(x.T, y.T, fclass)

    def plot(self, edges=False, show=False):
        rn = nx.draw_networkx_nodes(self, nodelist=self.active,
                                    pos=self.pos, node_size=_GRAPH_NODE_SIZE,
                                    node_color=_GRAPH_NODE_COLOR)
        if edges:
            re = nx.draw_networkx_edges(self, pos=self.pos, arrows=False)
            return (rn, re)
        else:
            return rn
        if show:
            plt.show()
