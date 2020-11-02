"""
Script for analyzing an SPDF measurement consisting of multiple frames, such
as obtained using non-destructive imaging.
"""

import numpy as np
import matplotlib.pyplot as plt
from bec4lib import *
from bec4fit import *

# This block of parameters (might) need to be changed between scans
imageIDs = np.arange(390355, 390444 + 1)
varname = "Generic_Hold_Time"
useConstrainedFits = True

rawimgs = queryImages(imageIDs)
camdat = queryImageSize(imageIDs)
varvals = queryVariable(imageIDs, varname)

imgdata = BEC4image(rawimgs, camdat)
imgdata.dispersiveImage([2, 3, 4], doPCA = True)

imshape = imgdata.pciImg.shape

# Find center using "all atom" shots, and cut out region around atoms
allimgs = imgdata.pciImg
(xp, yp) = findAtomPosition(allimgs[:, 0])
window = 15
allimgs = imgdata.pciImg[:, :, (yp - window) : (yp + window), (xp - window) : (xp + window)]

doublonMode = np.tile([1, 3, 2], imshape[0])
varvals = np.repeat(varvals, 3)

Ncounts = np.empty(imshape[:2] )
xwidths = np.empty( imshape[:2] )
ywidths = np.empty( imshape[:2] )
backgrounds = np.empty( imshape[:2] )

szs = allimgs.shape[2:]
X, Y = np.meshgrid( np.arange(szs[1]), np.arange(szs[0]) )

for i, img in enumerate(allimgs):
    for j, im in enumerate(img):
        fp = fit_2DGaussian(im)
        xwidths[i, j] = fp[0][2]
        ywidths[i, j] = fp[0][3]
        backgrounds[i, j] = fp[0][5]
        Ncounts[i, j] = 2*np.pi * np.abs( np.prod( fp[0][2:5] ) )

if useConstrainedFits:
    xw = np.mean(xwidths, axis = 0)
    yw = np.mean(ywidths, axis = 0)
    bckg = np.mean(backgrounds, axis = 0)
    
    for i, img in enumerate(allimgs):
        # Construct a fit function that has several parameters filled in
        fitfun = lambda X, x0, y0, a: gaussian2D(X[0], X[1], x0, y0, xw[j], yw[j], a, bckg[j])
        
        for j, im in enumerate(img):
            fp, _ = curve_fit(fitfun, (X.ravel(), Y.ravel()), im.ravel(), p0 = [14, 15, 0.6])
            Ncounts[i, j] = 2*np.pi * np.abs( xw[j] * yw[j] * fp[2] )
        
dblfrac, spdf, dblerr, spdferr = doublonAnalysis(Ncounts.ravel(), doublonMode, varvals)
univars = np.unique(varvals)

fig, ax = plt.subplots(1, 2, figsize = (12, 4))
ax[0].plot(varvals[::3], Ncounts[:, 0], 'o', label = 'all')
ax[0].plot(varvals[::3], Ncounts[:, 2], 'o', label = 'rm doublons')
ax[0].plot(varvals[::3], Ncounts[:, 1], 'o', label = 'rm pairs')
ax[0].legend(fontsize = 12)
ax[0].tick_params(labelsize = 12)
ax[0].set_xlabel(varname, fontsize = 14)

ax[1].errorbar(np.unique(varvals), spdf, spdferr, marker = 'o', ls = 'none', label = 'SPDF')
ax[1].errorbar(np.unique(varvals), dblfrac, dblerr, marker = 'o', ls = 'none', label = 'Doublon fraction')
ax[1].legend(fontsize = 12)
ax[1].tick_params(labelsize = 12)
ax[1].set_xlabel(varname, fontsize = 14)