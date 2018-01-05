import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

np.random.seed(123)

def plot_clusters(x,y,a,b,t):

        d1_min = np.min([x[:,0], y[:,0]])
        d1_max = np.max([x[:,0], y[:,0]])

        d2_D1min = (-a[0]*d1_min + b ) / a[1]
        d2_D1max = (-a[0]*d1_max + b ) / a[1]

        supp = float(t / a[1])
        plt.scatter(x[:,0], x[:,1], s=20)
        plt.scatter(y[:,0], y[:,1], s=20)

        plt.plot([d1_min, d1_max], [d2_D1min[0,0], d2_D1max[0,0]],color='k')
        plt.fill_between([d1_min, d1_max],
                        [d2_D1min[0,0] + supp, d2_D1max[0,0] + supp],
                        [d2_D1min[0,0] - supp, d2_D1max[0,0] - supp],
                        alpha=0.5, facecolor='0.75')

        plt.show()


def make_data(n):

        x_center = [1,1]
        y_center = [3,1]

        orientation_x = np.random.rand(2,2)
        orientation_y = np.random.rand(2,2)

        rx = np.clip(np.random.randn(n,2),-2,2)
        ry = np.clip(np.random.randn(n,2),-2,2)
        x = x_center + np.dot(rx, orientation_x)
        y = y_center + np.dot(ry, orientation_y)

        return x, y


if __name__ == '__main__':

        x, y = make_data(50)

        a = Variable(2)
        b = Variable()
        t = Variable()

        obj = Maximize(t)

        x_constraints = [a.T * x[i] - b >= t  for i in range(50)]
        y_constraints = [a.T * y[i] - b <= -t for i in range(50)]

        constraints = x_constraints +  y_constraints + [norm(a, 2) <= 1]

        prob = Problem(obj, constraints)
        prob.solve()

        plot_clusters(x, y, a.value, b.value, t.value)


