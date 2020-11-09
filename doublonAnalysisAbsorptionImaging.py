"""
Script for analyzing an SPDF measurement with kinetics absorption imaging
"""

import bec4fit
import bec4lib
import numpy as np
import matplotlib.pyplot as plt

ids = np.arange(392777, 392947 + 1)

imgs = bec4lib.queryImages(ids)
camInfo = bec4lib.queryImageSize(ids)
imgs = np.vstack(imgs)

dat = bec4lib.BEC4image(imgs,camInfo)
dat.absorptiveKinetic(knifeEdge=25, bottomEdge=30, doPCA=False, isFastKinetic = True)
scan_var = bec4lib.queryVariable(ids,'Generic_Hold_Time')
doublonMode = bec4lib.queryVariable(ids,varname='doublonMode')

# apply PCA by unique x-value groups
xvar_unique, key, counts = np.unique(scan_var,return_inverse=True,return_counts=True)

pwa = dat.pwa
pwoa = dat.pwoa
dark = dat.dark
temp = []
total = 0

for cts in counts:
    pwa_select = pwa[total:total+cts].reshape(cts,-1)
    pwoa_select = pwoa[total:total+cts].reshape(cts,-1)
    dark_select = dark[total:total+cts].reshape(cts,-1)

    total += cts
    pwoaTrue = pwoa_select-dark_select
    pwoaTrueMean = np.mean(pwoaTrue,axis=0)
    pwaTrue = pwa_select-dark_select

    _,_,VH = np.linalg.svd(pwoaTrue-pwoaTrueMean,full_matrices=False)
    estPWOA = ((pwaTrue-pwoaTrueMean)@(VH.T))@VH + pwoaTrueMean
    temp.append(-np.log(np.maximum(np.abs(pwaTrue/estPWOA),0.002)))

absImg_pca = np.vstack(temp).reshape(dat.shotsN,dat.colN,-1)

# calculate mean images per (doublonMode,scan_var) and obtain good fit guesses

constraints = dict()
mean_images = dict()

for xval in xvar_unique:
    for i in range(3):
        test_mean_image = np.mean(absImg_pca[np.logical_and(doublonMode==i+1,scan_var == xval)],axis=0)
        fparsx,fparsy,xcut,ycut = bec4fit.absImgNcount(test_mean_image)
        constraints[(i+1,xval)] = (fparsx,fparsy)
        mean_images[(i+1,xval)] = (xcut,ycut)

ncount = np.zeros((dat.shotsN))
for i,image in enumerate(absImg_pca):
    dm = doublonMode[i]
    xval = scan_var[i]
    fparsx,fparsy,_,_ = bec4fit.absImgNcount(image,isConstrained=True,p0c=constraints[dm,xval])
    nx = np.sqrt(2*np.pi)*fparsx[0]*np.abs(fparsx[2])
    ny = np.sqrt(2*np.pi)*fparsy[0]*np.abs(fparsy[2])
    ncount[i] = np.sqrt(nx*ny)

dblfrac,spdf,dblerr,spdferr = bec4lib.doublonAnalysis(ncount,doublonMode,scan_var)

fig = plt.figure(figsize=(12,4))
fig.add_subplot(1,2,1)
plt.errorbar(xvar_unique,spdf,yerr=spdferr,fmt="o",label='spdf')
plt.errorbar(xvar_unique,dblfrac,yerr=dblerr,fmt="o",label='dbl')
plt.legend()

fig.add_subplot(1,2,2)
plt.scatter(scan_var[doublonMode==1],ncount[doublonMode==1],label="all")
plt.scatter(scan_var[doublonMode==2],ncount[doublonMode==2],label="kill pairs")
plt.scatter(scan_var[doublonMode==3],ncount[doublonMode==3],label="kill doublons")
plt.ylim([np.min(ncount)/2,np.max(ncount)*1.1])
plt.legend()
#fig.suptitle("PCA per unique-x-value groups\n 35/13/35 positive-u pair, no fit constraint",fontsize=20)
#plt.savefig("back_to_abs_img.png")