import util
import random
import copy
import cPickle
import copy_reg
import types
import curses
import numpy as np
import gp.core as gp
import gp.kernels as kernels
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d

# Constants
_PAUSE_ON = False
_LT_COLOR = '#1F6E99'
_UT_COLOR = '#F2A230'
_VT_COLOR = '#DDDDDD'
_CWP_COLOR = '#00DD00'
_PATH_COLOR = '#C3D9C3'
_XLIM_SLACK = 50
_YLIM_SLACK = 1
_NGRID = 200
_BETA = 3


# Pickle methods by name (used to circumvent problem
# with pickling of instance methods)
def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))

copy_reg.pickle(types.MethodType, reduce_method)

class Planner(object):
    def __init__(self, tc, rule='miu_dp', spe=10, nla=10):
        self.tc = tc
        self.rule = rule
        self.spe = spe
        self.nla = nla
        (self.gx1, self.gx2, _) = tc.grid(_NGRID)
        self.gfx = np.hstack((self.gx1.reshape(-1, 1), self.gx2.reshape(-1, 1)))
        self.init_model()
        self.graph = util.Graph.from_testcase(tc, self.fclass)
        self.reset()

    def fclass(self, x):
        (m, v) = self.model.inf(x)
        unclass = np.logical_and((m - _BETA * np.sqrt(v)).flat < self.tc.h,
                                 (m + _BETA * np.sqrt(v)).flat > self.tc.h)
        return list(np.arange(1, len(unclass)+1)[unclass])

    def init_model(self):
        hyp = {'mean': 2.4, 'cov': [5.5, 1, 0.75], 'lik': -1}
        kernel = kernels.SE(hyp)
        self.model = gp.GP(kernel)

    def reset(self):
        self.cwp = self.graph.init_node()
        self.fullpath = [self.cwp]
        self.path = []
        self.final = False
        self.model.clear()
        self.ut = set(range(_NGRID*_NGRID))
        self.ht = set()
        self.lt = set()

    def sample_edge(self, e):
        (v1, v2) = e
        xs = np.linspace(self.graph.pos[v1][0],
                         self.graph.pos[v2][0],
                         self.spe)
        ys = np.linspace(self.graph.pos[v1][1],
                         self.graph.pos[v2][1],
                         self.spe)
        x = np.vstack((xs, ys)).T
        return x[1:,:]

    def eval_edge(self, e, mode='miu'):
        x = self.sample_edge(e)
        if mode == 'mi':
            return self.model.minfo(x)
        elif mode == 'miu':
            (m, v) = self.model.inf(x)
            unclass = np.logical_and((m - _BETA * np.sqrt(v)).flat < self.tc.h,
                                     (m + _BETA * np.sqrt(v)).flat > self.tc.h)
            return self.model.minfo(x[unclass,:])
        else:
            raise Exception('Invalid edge evaluation mode')

    def eval_path(self, p, mode='miu'):
        x = np.zeros((0, 2))
        edges = zip(p[:-1], p[1:])
        for e in edges:
            x = np.vstack((x, self.sample_edge(e)))
        if mode == 'mi':
            return self.model.minfo(x)
        elif mode == 'miu':
            (m, v) = self.model.inf(x)
            unclass = np.logical_and((m - _BETA * np.sqrt(v)).flat < self.tc.h,
                                     (m + _BETA * np.sqrt(v)).flat > self.tc.h)
            return self.model.minfo(x[unclass,:])
        else:
            raise Exception('Invalid path evaluation mode')

    def dp(self, v, n):
        nodes = [[(v, 0, [v])]] # nodeid, score, path
        for lvl in range(1, n+1):
            nextcol = self.graph.pcolumn(nodes[-1][0][0])
            nodes.append([])
            for u in nextcol:
                nodes[-1].append((u, None, None, None))
        for lvl in range(1, n+1):
            for (i, node) in enumerate(nodes[lvl]):
                u = node[0]
                sc = []
                for pu, psc, ppath in nodes[lvl-1]:
                    sc.append(self.eval_path(ppath + [u], 'miu'))
                maxi = np.argmax(sc)
                nodes[lvl][i] = (u, sc[maxi], nodes[lvl-1][maxi][2] + [u])
        maxidx = max(xrange(len(nodes[-1])), key=lambda i: nodes[-1][i][1])
        return nodes[-1][maxidx][2]
        
    def get_path(self):
        if self.rule == 'rand_1':
            self.path = [self.cwp, random.choice(self.graph.pcolumn(self.cwp))]
        elif self.rule == 'mi_1':
            col = self.graph.pcolumn(self.cwp)
            idx = np.argmax([self.eval_edge((self.cwp, v), 'mi') for v in col])
            self.path = [self.cwp, col[idx]]
        elif self.rule == 'miu_1':
            col = self.graph.pcolumn(self.cwp)
            idx = np.argmax([self.eval_edge((self.cwp, v), 'miu') for v in col])
            self.path = [self.cwp, col[idx]]
        elif self.rule == 'rand_5':
            self.path = self.graph.rand_path(self.cwp)
        elif self.rule == 'miu_best_rand_5':
            maxsc = -1
            for i in xrange(100):
                p = self.graph.rand_path(self.cwp)
                sc = self.eval_path(p, 'miu')
                if sc > maxsc:
                    maxsc = sc
                    self.path = p
        elif self.rule == 'miu_dp':
            self.path = self.dp(self.cwp, self.curnla)
        else:
            raise Exception('Invalid planning rule')

    def classify(self):
        (m, v) = self.model.inf(self.gfx)
        n = _NGRID*_NGRID
        self.ht = set(np.arange(n)[(m - _BETA * np.sqrt(v)).flat > self.tc.h])
        self.lt = set(np.arange(n)[(m + _BETA * np.sqrt(v)).flat < self.tc.h])
        self.ut = set(np.arange(n)) - self.ht - self.lt

    def run(self, npasses=2, record=False, recfilepath=None):
        if record:
            rec = Recorder(self)
        plt.ion()
        self.reset()
        self.curnla = min(self.nla, npasses*(self.graph.xres-1))
        np = 0
        while np < npasses:
            self.graph.update_active([self.cwp])
            self.get_path()
            self.plot()
            if record:
                rec.record(self)
            if _PAUSE_ON:
                raw_input('')
            x = self.sample_edge((self.cwp, self.path[1]))
            y = self.tc.sample(x)
            self.model.add(x, y)
            self.classify()
            self.cwp = self.path[1]
            self.fullpath.append(self.cwp)
            # Check for final pass
            if self.graph.is_terminal(self.path[-1]):
                term = [i for (i, p) in enumerate(self.path[1:]) if
                        self.graph.is_terminal(p)]
                if len(term) == npasses - np:
                    self.final = True
            if self.final:
                self.curnla = self.curnla - 1
            if self.graph.is_terminal(self.cwp):
                np = np + 1
        self.path = [self.cwp]
        self.plot()
        if record:
            rec.record(self)
            rec.end(self)
            rec.save(recfilepath)
        if _PAUSE_ON:
            raw_input('')
        if record:
            return rec

    def plot(self):
        plt.clf()
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
        plt.xlim(self.tc.lim['x1'][0]-_XLIM_SLACK,
                 self.tc.lim['x1'][1]+_XLIM_SLACK)
        plt.ylim(self.tc.lim['x2'][0]-_YLIM_SLACK,
                 self.tc.lim['x2'][1]+_YLIM_SLACK)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        self.graph.plot()
        y = np.zeros((_NGRID, _NGRID))
        if self.ht:
            y[np.unravel_index(list(self.ht), y.shape)] = 1
        if self.lt:
            y[np.unravel_index(list(self.lt), y.shape)] = -1
        cmap = plt.cm.get_cmap('gray')
        cmap.set_under(_LT_COLOR)
        cmap.set_over(_UT_COLOR)
        plt.contourf(np.asarray(self.gx1), np.asarray(self.gx2),
                     np.asarray(y), cmap=cmap, extend='both',
                     levels=[-0.0001, 0])
        edgelist = zip(self.path[:-1], self.path[1:])
        nx.draw_networkx_edges(self.graph, self.graph.pos, edgelist=edgelist,
                               arrows=False, edge_color=_PATH_COLOR,
                               alpha=0.5, width=3)
        plt.plot(self.model.x[:,0], self.model.x[:,1], 'o',
                 markerfacecolor=_VT_COLOR,
                 markeredgecolor=_VT_COLOR,
                 alpha=0.5,
                 markersize=3)
        plt.plot(self.graph.pos[self.cwp][0],
                 self.graph.pos[self.cwp][1], 'o',
                 markerfacecolor=_CWP_COLOR,
                 markeredgecolor='k',
                 alpha=0.7,
                 markersize=20)
        plt.draw()

_KEY_Q = 113
class Recorder(object):
    def __init__(self, base):
        self.base = copy.deepcopy(base)
        self.active = []
        self.path = []

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'r') as f:
            return cPickle.load(f)

    def record(self, obj):
        self.path.append(obj.path)
        self.active.append(obj.graph.active)

    def end(self, obj):
        self.x = obj.model.x
        self.y = obj.model.y

    def plot(self, t):
        self.base.cwp = self.path[t][0]
        self.base.path = self.path[t]
        self.base.model.clear()
        self.base.model.add(self.x[:(self.base.spe-1)*t, :],
                            self.y[:(self.base.spe-1)*t])
        self.base.classify()
        self.base.graph.active = self.active[t]
        self.base.plot()

    def replay(self):
        plt.ion()
        t = 0
        win = curses.initscr()
        self.win = win
        win.keypad(1)
        win.clear()
        self.plot(t)
        while True:
            newt = t
            s = win.getch()
            if s == _KEY_Q:
                curses.endwin()
                return
            elif s == curses.KEY_RIGHT:
                newt = min(len(self.path)-1, t+1)
            elif s == curses.KEY_LEFT:
                newt = max(0, t-1)
            if newt != t:
                t = newt
                self.plot(t)

    def save(self, fpath):
        if not fpath:
            fpath = 'log'
        with open(fpath, 'w') as f:
            cPickle.dump(self, f)
