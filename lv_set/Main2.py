from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
import drlse_algo as drlse
import drlse_algo2 as drlse2
import numpy as np
import skimage.external.tifffile as tiffy

img = np.array(imread('gourd.bmp', True), dtype='float32')
img2 = tiffy.imread('OneSphere.tiff')
ones = np.ones_like(img2)
img2=ones-img2
img2 = np.round(img2[145-50:145+50, 145-50:145+50, 145-50:145+50]*255)

#%% parameters 3D
timestep = 1        # time step
mu = 0.2/timestep   # coefficient of the distance regularization term R(phi)
iter_inner = 4
iter_outer = 25
lmda = 2            # coefficient of the weighted length term L(phi)
alfa = -9           # coefficient of the weighted area term A(phi)
epsilon = 2.0       # parameter that specifies the width of the DiracDelta function
sigma = 0.8         # scale parameter in Gaussian kernel

#%% parameters 3D
timestep2 = 1        # time step
mu2 = 1/timestep2   # coefficient of the distance regularization term R(phi)
iter_inner2 = 4
iter_outer2 = 25
lmda2 = 2            # coefficient of the weighted length term L(phi)
alfa2 = -9           # coefficient of the weighted area term A(phi)
epsilon2 = 2.0       # parameter that specifies the width of the DiracDelta function
sigma2 = 0.8         # scale parameter in Gaussian kernel

#%% Image editing
img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
img_smooth2 = filters.gaussian_filter(img2, sigma)    # smooth image by Gaussian convolution

#%% Analysis

[Iy, Ix] = np.gradient(img_smooth)
[Iz2, Iy2, Ix2] = np.gradient(img_smooth2)

f = np.square(Ix) + np.square(Iy)
f2 = np.square(Ix2) + np.square(Iy2) + np.square(Iz2)

g = 1 / (1+f)
g2 = 1 / (1+f2)

c0 = 2
initialLSF = c0 * np.ones(img.shape)
initialLSF2 = c0 * np.ones(img2.shape)

print(initialLSF.shape)
print(initialLSF2.shape)

initialLSF[24:35, 20:26] = -c0
initialLSF2[45:55, 45:55, 45:55] = -c0

phi = initialLSF.copy()
phi2 = initialLSF2.copy()

plt.ion()
#%%

contours = measure.find_contours(phi, 0)
phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner)
contours = measure.find_contours(phi, 0)
plt.plot(contours[0][:,1],contours[0][:,0], 'k')
plt.imshow(phi)

# %% Plotting definitions 2D - #%% DRLSE
fig = plt.figure(1)
def show_fig():
    contours = measure.find_contours(phi, 0)
    ax = fig.add_subplot(111)
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()

# start level set evolution
for n in range(iter_outer):
    phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner)
    if np.mod(n, 2) == 0:
        print('fig 1 for %i time' % n)
        fig.clf()
        show_fig()
        plt.pause(0.3)

alfa = 0
iter_refine = 10
phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine)
finalLSF = phi.copy()
print('show final fig 1')
fig.clf()
show_fig()
plt.pause(2)
plt.show()


#%% parameters 3D
timestep2 = 1        # time step
mu2 = 0.2/timestep2   # coefficient of the distance regularization term R(phi)
iter_inner2 = 4
iter_outer2 = 25
lmda2 = 2            # coefficient of the weighted length term L(phi)
alfa2 = -9           # coefficient of the weighted area term A(phi)
epsilon2 = 2       # parameter that specifies the width of the DiracDelta function
sigma2 = 0.1         # scale parameter in Gaussian kernel

#%%
phi2 = initialLSF2.copy()
contours2 = measure.find_contours(phi2[phi2.shape[0]//2], 0)
#plt.imshow(img2[img2.shape[0]//2], cmap="Greys_r")
#plt.plot(contours2[0][:,0],contours2[0][:,1])

#%% Trials

phi2 = drlse2.drlse_edge(phi2, g2, lmda2, mu2, alfa2, epsilon2, timestep2, iter_inner2)
contours2 = measure.find_contours(phi2[phi2.shape[0]//2], 0)

#plt.imshow(img2[img2.shape[0]//2], cmap="Greys_r")
plt.plot(contours2[0][:,1],contours2[0][:,0], 'k')

#plt.figure()
plt.imshow(phi2[phi2.shape[0]//2])

# %% Plotting definitions 3D - DRLSE 3D
fig2 = plt.figure(2)
def show_fig2():
    contours2 = measure.find_contours(phi2[phi2.shape[0]//2], 0)
    ax2 = fig.add_subplot(111)
    ax2.imshow(img2[img2.shape[0]//2], interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours2):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()

# start level set evolution
for n in range(iter_outer2):
    phi2 = drlse2.drlse_edge(phi2, g2, lmda2, mu2, alfa2, epsilon2, timestep2, iter_inner2)
    if np.mod(n, 2) == 0:
        print('fig 2 for %i time' % n)
        fig.clf()
        show_fig2()
        plt.pause(0.3)

alfa2 = 0
iter_refine2 = 10
phi2 = drlse2.drlse_edge(phi2, g2, lmda2, mu2, alfa2, epsilon2, timestep2, iter_refine2)
finalLSF2 = phi2.copy()
print('show final fig 2')
fig.clf()
show_fig()
plt.pause(2)
plt.show()



