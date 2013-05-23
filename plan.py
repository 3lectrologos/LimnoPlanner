import util
import random
import numpy as np
import gp.core as gp
import gp.kernels as kernels
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d


def test():
    import util
    tc = util.test2()
    p = Planner(tc)
    return p

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

class Planner(object):
    def __init__(self, tc, rule='miu_dp', spe=10, nla=10):
        self.tc = tc
        self.rule = rule
        self.spe = spe
        self.nla = nla
        self.graph = util.Graph.from_testcase(tc)
        (self.gx1, self.gx2, _) = tc.grid(_NGRID)
        self.gfx = np.hstack((self.gx1.reshape(-1, 1), self.gx2.reshape(-1, 1)))
        self.init_model()
        self.reset()

    def init_model(self):
        hyp = {'mean': 2.4, 'cov': [5.5, 1, 0.75], 'lik': -1}
        kernel = kernels.SE(hyp)
        self.model = gp.GP(kernel)

    def reset(self):
        self.np = 0
        self.cwp = 1
        self.path = []
        self.final = False
        self.model.clear()
        self.ut = set(range(_NGRID*_NGRID))
        self.ht = set()
        self.lt = set()
        self.vt = np.zeros((0, 2))
        self.torem = []
        plt.figure()
        self.graph.plot()
        plt.xlim(self.tc.lim['x1'][0]-_XLIM_SLACK,
                 self.tc.lim['x1'][1]+_XLIM_SLACK)
        plt.ylim(self.tc.lim['x2'][0]-_YLIM_SLACK,
                 self.tc.lim['x2'][1]+_YLIM_SLACK)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

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

    def eval_edge(self, e, mode='miu', xout=False):
        x = self.sample_edge(e)
        if mode == 'mi':
            if xout:
                return (self.model.minfo(x), x)
            else:
                return self.model.minfo(x)
        elif mode == 'miu':
            (m, v) = self.model.inf(x)
            unclass = np.logical_and((m - _BETA * np.sqrt(v)).flat < self.tc.h,
                                     (m + _BETA * np.sqrt(v)).flat > self.tc.h)
            if xout:
                return (self.model.minfo(x[unclass,:]), x[unclass,:])
            else:
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
            nextcol = self.graph.column(nodes[-1][0][0])
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
            self.path = [self.cwp, random.choice(self.graph.column(self.cwp))]
        elif self.rule == 'mi_1':
            col = self.graph.column(self.cwp)
            idx = np.argmax([self.eval_edge((self.cwp, v), 'mi') for v in col])
            self.path = [self.cwp, col[idx]]
        elif self.rule == 'miu_1':
            col = self.graph.column(self.cwp)
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
            self.path = self.dp(self.cwp, self.nla)
        else:
            raise Exception('Invalid planning rule')

    def check_final(self):
        pass
    
    def classify(self):
        (m, v) = self.model.inf(self.gfx)
        n = _NGRID*_NGRID
        self.ht = set(np.arange(n)[(m - _BETA * np.sqrt(v)).flat > self.tc.h])
        self.lt = set(np.arange(n)[(m + _BETA * np.sqrt(v)).flat < self.tc.h])
        self.ut = set(np.arange(n)) - self.ht - self.lt

    def run(self, npasses=2):
        plt.ion()
        self.reset()
        while self.np < npasses:
            self.get_path()
            self.plot()
            self.check_final()
            x = self.sample_edge((self.cwp, self.path[1]))
            y = self.tc.sample(x)
            self.model.add(x, y)
            self.classify()
            self.vt = np.vstack((self.vt, x))
            self.cwp = self.path[1]
            if self.graph.is_terminal(self.cwp):
                self.np = self.np + 1
            if _PAUSE_ON:
                raw_input('')
        self.plot()
        if _PAUSE_ON:
            raw_input('')

    def rem(self, arg):
        self.torem.extend(arg)

    def remplt(self):
        [arg.remove() for arg in self.torem]
        self.torem = []
        
    def plot(self):
        self.remplt()
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif')
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
        #plt.contour(np.asarray(self.gx1), np.asarray(self.gx2), np.asarray(y),
        #            colors='k', linestyles='solid', levels=[-1, 0, 1])
        edgelist = zip(self.path[:-1], self.path[1:])
        nx.draw_networkx_edges(self.graph, self.graph.pos, edgelist=edgelist,
                               arrows=False, edge_color=_PATH_COLOR,
                               alpha=0.5, width=3)
        self.rem(plt.plot(self.vt[:,0], self.vt[:,1], 'o',
                          markerfacecolor=_VT_COLOR,
                          markeredgecolor=_VT_COLOR,
                          alpha=0.5,
                          markersize=3))
        self.rem(plt.plot(self.graph.pos[self.cwp][0],
                          self.graph.pos[self.cwp][1], 'o',
                          markerfacecolor=_CWP_COLOR,
                          markeredgecolor='k',
                          alpha=0.7,
                          markersize=20))
        plt.draw()
