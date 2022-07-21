import numpy as np
from matplotlib import pyplot as plt
import time
import sys
#import seaborn as sns

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def BA_transition(eps, z):
    #  Ballistic annihilation transition map for esp amount of time
    #  input Z is 3 by N array
    #  Z[0,i] is the location of particle i
    #  Z[1,i] is the speed of particle i
    #  z[2,i] is the indicator of particle being alive
    #  two particles annihilate if they are within distance eps
    z_new = z.copy()
    z_new[0,:] = z[0,:] + z[1,:]*eps

    #  delete two live particles within distance 2*eps
    #  this could be improved by sorting and considering nearest particles
    I = indices(z[2,:], lambda x: x == 1)
    for i in I:
        for j in I:
            if np.absolute(z_new[0, i] - z_new[0, j]) < 2*eps and i<j:
                #  np.delete(z_new,i,1) # delete ith column of z_new
                z_new[2, i] = 0
                z_new[2, j] = 0

    return z_new


def BA_ini(p0, q, v, N):
    #  Constructs an initial configuration z for BA
    #  Density of speed 0 = p0, v = (1-q)(1-p0), -1 = q(1-p0)
    #  Z[0,i] is the location of particle i
    #  Z[1,i] is the speed of particle i
    #  z[2,i] is the indicator of particle being alive
    #  Particle of index N is conditioned to be a blockade

    # Poisson Point Process of N points
    E = np.random.exponential(1, 2 * N)  # initial gaps between particles
    x = np.array([])
    for i in np.arange(2 * N + 1):
        x = np.hstack((x, sum(E[0:i])))

    # sample speed configuration
    s = np.random.choice(3, 2 * N + 1, p=[(1 - q) * (1 - p0), p0, q * (1 - p0)])
    s = s - 1
    s[s == -1] = -v
    s[N] = 0  # condition on the middle particle of index N to have speed 0

    # construct initial configuration
    z = np.vstack((x, s, np.ones((1, 2 * N + 1))))

    return z


def BA_trj(p0, q, v, N, time, eps):
    #  Ballistic annihilation with speeds -v,0,1
    #  Density of speed 0 = p0, v = (1-q)(1-p0), -1 = q(1-p0)
    #  2*N+1 = number of initial particles
    #  eps = duration of one iteration

    # build BA trajectory
    z = BA_ini(p0, q, v, N)
    trj = z
    for k in np.arange(time):
        z = BA_transition(eps, z)
        trj = np.dstack((trj, z))

    return trj


def BA_plot(p0, q, v, N, time, eps):
    trj = BA_trj(p0, q, v, N, time, eps)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x1 = trj[0, :, :]  # x1(i,j) = location of particle i at iteration j
    y1 = np.arange(time+1)
    for i in np.arange(2*N):
        y1 = np.vstack((y1, np.arange(time + 1)))
    s1 = trj[1, :, :]  # s1(i,j) = velocity of particle i at iteration j
    w1 = trj[2, :, :]  # w1(i,j) = indicator of whether particle i is alive at iteration j

    x1 = x1.flatten()  # spatial coordinate of particles
    y1 = y1.flatten()  # temporal coordinate of particles
    s1 = s1.flatten()  # velocity of particles
    w1 = w1.flatten()  # indicator of live particles
    J = (w1 == 1)

    x = x1[J]
    y = eps*y1[J]
    s = s1[J]
    rgb = plt.get_cmap('jet')(s+1)
    ax.scatter(x, y, color=rgb, s=0.0002)
    plt.xlim(0, 2*N+1)
    plt.ylim(0, 300)
    plt.title('p=%1.2f, lambda_left=%1.2f, v=%1.2f, #particles=%i' % (p0, q, v, N))
    plt.show()

    z = np.vstack((x, y))
    return z


def ind_hit_both(z, N, steps, eps):
    #  z = (3 by 2*N+1) BA configuration with blockade at N
    #  Function gives 1 if the blockade of index N is visited by both L and R by time 'steps'
    #  Function gives 0 otherwise

    left = 0  # indicator of left hit
    right = 0  # indicator of right hit
    z_new = z.copy()
    for i in np.arange(steps):
        I = indices(z[2, :], lambda x: x == 1)  # indices of alive particles
        for j in I:
            if 0 < z_new[0, j] - z_new[0, N] < 2 * eps:
                left = 1
            if 0 < z_new[0, N] - z_new[0, j] < 2 * eps:
                right = 1
        z_new = BA_transition(eps, z_new)

    ind = 0
    if left > 0 and right > 0:
        ind = 1

    return ind


def ind_hit_whichside(z, N, steps, eps):
    #  z = (3 by 2*N+1) BA configuration with blockade at N
    #  Function gives -1 if the blockade at N is first hit from right at time m<'steps'
    #  Function gives +1 if the blockade at N is first hit from left at time m<'steps'
    #  Function gives 0 if the blockade at N is alive at time 'steps'

    ind = 0  # default output value

    # build BA trajectory
    trj = z
    for k in np.arange(steps):
        z = BA_transition(eps, z)
        trj = np.dstack((trj, z))

    w = trj[2, :, :]  # w[i,j] = indicator of whether particle i is alive at iteration j
    w1 = w[N, :]  # w1[j] = indicator of whether particle N is alive at iteration j
    j1 = indices(w1, lambda x: x == 1)
    m = len(j1)  # last time before time 'steps' that the blockade of index N is alive

    if m < steps:
        I = indices(w[:, m-1], lambda x: x == 1)
        for i in I:
            if 0 < trj[0, i, m] - trj[0, N, m] < 2 * eps:
                ind = -1  # hit by left particle
            if 0 < trj[0, N, m] - trj[0, i, m] < 2 * eps:
                ind = 1  # hit by right particle

    return ind


def alpha(p0, q, v, N, steps, eps):
    #  approximates \cev{\alpha}(p,\lambda,v)
    c = 0
    d = 0
    bar = progressbar.ProgressBar()
    for step in bar(range(steps)):
        z = BA_ini(p0, q, v, N)  # initial configuration
        if ind_hit_both(z, N, np.floor(N/eps), eps) > 0:
            d = d+1
            if ind_hit_whichside(z, N, np.floor(N/eps), eps) < 0:
                c = c+1

    return c / d  # empirical frequency contributing to \alpha


def alpha_plot(q, v, N, steps, eps, mesh):
    # plot of \cev{\alpha}(p0, q, v)
    # draw figure
    # sns.set_style('darkgrid', {'legend.frameon': True})
    x = np.linspace(0, 1/2, num=mesh)
    y = np.array([])
    for i in np.arange(mesh):
        y = np.hstack((y, alpha(x[i], q, v, N, steps, eps)))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, s=3, c='b', marker="s", label='')
    plt.xlabel('p')
    plt.ylabel('cev(alpha)')
    plt.title('lambda=%1.3f, v=%1.1f' % (q, v))
    plt.axis([0, 1, 0, 1])
    plt.legend()

    return y

BA_plot(5/18 - 0.01, 1/2, 2, 2000, 1000, 0.3)

BA_plot(5/18 + 0.01, 1/2, 2, 2000, 1000, 0.3)