import numpy as np
import scipy.ndimage.filters as filters


def drlse_edge(phi_0, g, lmda, mu, alfa, epsilon, timestep, iters):  # Updated Level Set Function
    """
    :param phi_0: level set function to be updated by level set evolution
    :param g: edge indicator function
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param timestep: time step
    :param iters: number of iterations
    :param potentialFunction: choice of potential function in distance regularization term.
    """

    phi = phi_0.copy()
    [vz, vy, vx] = np.gradient(g)
    for k in range(iters):
        #phi = NeumannBoundCond(phi)
        [phi_z, phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y) + np.square(phi_z))
        smallNumber = 1e-100
        Nx = phi_x / (s + smallNumber)  # add a small positive number to avoid division by zero
        Ny = phi_y / (s + smallNumber)
        Nz = phi_z / (s + smallNumber)
        curvature = div(Nx, Ny, Nz)
        distRegTerm = distReg_p2(phi)
        diracPhi = Dirac(phi, epsilon)
        areaTerm = diracPhi * g  # balloon/pressure force
        edgeTerm = diracPhi * (vx * Nx + vy * Ny + vz * Nz) + diracPhi * g * curvature
        phi = phi + timestep * (mu * distRegTerm + lmda * edgeTerm + alfa * areaTerm)
    return phi


def distReg_p2(phi):
    [phi_z, phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y) + np.square(phi_z))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)  
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))  
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y, dps * phi_z - phi_z) + filters.laplace(phi, mode='wrap')


def div(nx, ny, nz):
    [junk, junk, nxx] = np.gradient(nx)
    [junk, nyy, junk] = np.gradient(ny)
    [nzz, junk, junk] = np.gradient(nz)
    return nxx + nyy + nzz


def Dirac(x, sigma):
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def NeumannBoundCond(f):
    """
        Make a function satisfy Neumann boundary condition
    """
    [nz, ny, nx] = f.shape
    g = f.copy()

#    g[0, 0, 0] = g[2, 2, 2]
#    g[0, 0, nx-1] = g[0, 2, nx-3]
#    g[:, ny-1, 0] = g[:, ny-3, 2]
#    g[:, ny-1, nx-1] = g[:, ny-3, nx-3]
#
#    g[:, 0, 1:-1] = g[:, 2, 1:-1]
#    g[:, ny-1, 1:-1] = g[:, ny-3, 1:-1]
#
#    g[:, 1:-1, 0] = g[:, 1:-1, 2]
#    g[:, 1:-1, nx-1] = g[:, 1:-1, nx-3]

    return g
