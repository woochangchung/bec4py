import numpy as np
import pymysql.cursors
from scipy.optimize import curve_fit

def server_settings():
    # Returns our server settings; can be passed into pymysql using dictionary splat: **
    return dict(host = "18.62.27.10",
         user = "root",
         password = "w0lfg4ng",
         db = "becivdatabase",
         cursorclass = pymysql.cursors.DictCursor)

def queryImages(ids):
    # Grab raw images from the database. Returns an array with 1D images, so they
    # need to be reshaped by using the image size.
    try:
        conn = pymysql.connect(**server_settings())
        if type(ids) != np.ndarray:
            # If there's only one image we're interested in we need different formatting
            imageIDstring = str(ids)
        else:
            imageIDstring = ', '.join(['%i' % s for s in ids])
        with conn.cursor() as cursor:
            sql1 = "SELECT data FROM images WHERE imageID IN (" + imageIDstring + ")"
            cursor.execute(sql1)
            result = cursor.fetchall()
    finally:
        conn.close()
    
    imgs = np.vstack([np.frombuffer(ds['data'], dtype = 'int8').view('int16') for ds in result])
    return imgs

def queryImageSize(ids):
    # Grab the camera settings belonging to a particular image.
    try:
        conn = pymysql.connect(**server_settings())
        if type(ids) != np.ndarray:
            # If there's only one image we're interested in we need different formatting
            imageIDstring = str(ids)
        else:
            imageIDstring = ', '.join(['%i' % s for s in ids])
            
        with conn.cursor() as cursor:
            sql1 = "SELECT cameraID_fk FROM images WHERE imageID IN (" + imageIDstring + ");"
            cursor.execute(sql1)
            result = cursor.fetchall()
    finally:
        conn.close()
    
    # Get unique camera IDs, and obtain sizes
    ids = np.unique([ x['cameraID_fk'] for x in result ])
    try:
        conn = pymysql.connect(**server_settings())
        if len(ids) == 1:
            idstring = str(ids[0])
        else:
            idstring = ', '.join(['%i' % s for s in ids])
        
        with conn.cursor() as cursor:
            sql1 = "SELECT cameraHeight, cameraWidth, Depth FROM cameras WHERE cameraID IN (" + idstring + ");"
            cursor.execute(sql1)
            result = cursor.fetchall()
    finally:
        conn.close()
    return result

def queryVariable(ids, varname):
    # Grab a variable value belonging to a particular image
    try:
        conn = pymysql.connect(**server_settings())
        if type(ids) != np.ndarray:
            # If there's only one image we're interested in we need different formatting
            imageIDstring = str(ids)
        else:
            imageIDstring = ', '.join(['%i' % s for s in ids])
            
        with conn.cursor() as cursor:
            sql1 = "SELECT runID_fk FROM images WHERE imageID IN (" + imageIDstring + ");"
            cursor.execute(sql1)
            result = cursor.fetchall()
    finally:
        conn.close()
    
    # Get variable values
    ids = [x['runID_fk'] for x in result]
    try:
        conn = pymysql.connect(**server_settings())
        if len(ids) == 1:
            idstring = str(ids[0])
        else:
            idstring = ', '.join(['%i' % s for s in ids])
        
        with conn.cursor() as cursor:
            sql1 = "SELECT " + varname + " FROM ciceroout WHERE runID IN (" + idstring + ");"
            cursor.execute(sql1)
            result = cursor.fetchall()
    finally:
        conn.close()
    return np.array([float(x[varname]) for x in result])

class myError(Exception):
    """ 
    Image processing related exceptions
    """
    pass

class frameNumError(myError):
    """
    Raised when the number of frames is inappropriate for the used image processing
    """

class BEC4image:
    """
    Class for BEC4 image object.
    Initialize with 2D numpy array (shape: (num_images, bytelength)) and camera info (use 'queryImageSize()')
    """
    def __init__(self,rawdata,camInfo):
        self.rowN = camInfo[0]['cameraHeight']
        self.colN = camInfo[0]['cameraWidth']
        self.frameN = camInfo[0]['Depth']
        self.bytesLength = self.rowN*self.colN*self.frameN
        self.shotsN = rawdata.shape[0]
        self.raw = rawdata.reshape(-1,self.frameN,self.colN,self.rowN)
        self.pwa = None
        self.pwoa = None
        self.dark = None
    
    def absorptiveImage(self):
        try:
            if self.frameN != 3:
                raise frameNumError
            self.pwa = np.minimum(self.raw[:,0,:,:],65535)
            self.pwoa = np.minimum(self.raw[:,1,:,:],65535)
            self.dark = np.minimum(self.raw[:,2,:,:],65535)
            a_up = np.minimum(self.pwa-self.dark,65536,dtype='float')
            a_down = np.minimum(self.pwoa-self.dark,65536,dtype='float')
            a_down = np.maximum(a_down,1,dtype='float')
            self.absImg = -np.log(np.maximum(np.abs(a_up/a_down),0.002))
            return self.absImg
        except frameNumError:
            print(f"you only have {self.frameN} frame(s). You need three to compute normal absorption images")
    
    def absorptiveKinetic(self,knifeEdge=30, bottomEdge = 20, doPCA = False):
        """
        create absorption images assuming a single-shot kinetics image with three windows.

        knifeEdge: row index below which there is no razor blade visible

        bottomEdge: number of rows with artifacts
        """
        try:
            if self.frameN != 1:
                raise frameNumError
            windowSize = 127
            numWindows = self.rowN//windowSize
            
            self.pwa = np.minimum(self.raw[:,0,:,knifeEdge:windowSize-bottomEdge],65536,dtype='float')
            self.pwoa = np.minimum(self.raw[:,0,:,windowSize+knifeEdge:2*windowSize-bottomEdge],65536,dtype='float')
            self.dark = np.minimum(self.raw[:,0,:,2*windowSize+knifeEdge:3*windowSize-bottomEdge],65536,dtype='float')
            a_up = np.minimum(self.pwa-self.dark,65536,dtype='float')
            a_down = np.minimum(self.pwoa-self.dark,65536,dtype='float')
            a_down = np.maximum(a_down,1,dtype='float')
            self.absImg = -np.log(np.maximum(np.abs(a_up/a_down),0.002))

            if doPCA:
                a_down = a_down.reshape(self.shotsN,-1)
                a_up = a_up.reshape(self.shotsN,-1)

                meanPWOA = np.mean(a_down,axis=0)
                pwoa_flat = a_down - meanPWOA
                u,s,vh = np.linalg.svd(pwoa_flat,full_matrices=False)
                estPWOA = ((a_up-meanPWOA)@vh.T)@vh + meanPWOA
                estPWOA
                self.absImg = -np.log(np.maximum(np.abs(a_up/estPWOA),0.002)).reshape(self.shotsN,self.colN,windowSize-knifeEdge-bottomEdge)

        except frameNumError:
            print(f"you have {self.frameN} frames. That's too many for kinetics imaging.")

    def dispersiveImage(self,pwaLoc,knifeEdge = 20, doPCA = False):
        """
        inhomogeneously weighted expectation value of total atom number. See Lab book on 10/08/2020
        
        compute dispersive image for variable numbero of PWA windows.
        
        pwaLoc: list or other iterable of indices starting from one that denotes the location of PWA windows
        
            e.g. pwaLoc = [1,3,4]
            The first, third, and fourth windows are PWA images.
        
        knifeEdge: starting row number from which there should be no razor edge visible in the image
        
        doPCA: flag for enabling PCA 
        
        Assumes window size of 127 rows for kinetics imaging
        """
        try:
            if self.frameN !=1:
                raise frameNumError
            windowSize = 127
            numWindows = self.rowN//windowSize
            
            self.pwa = np.zeros((self.shotsN,len(pwaLoc),self.colN,windowSize-knifeEdge-1))
            self.pciImg = np.zeros((self.shotsN,len(pwaLoc),self.colN,windowSize-knifeEdge-1))
            self.darkGround = np.zeros((self.shotsN,len(pwaLoc),self.colN,windowSize-knifeEdge-1)) # estimate scattered power
            
            if doPCA:
                # Duplicate dark images such that the array has the same size as PW(O)A, so we can calculate the PCI image in one go below
                self.dark = np.minimum(self.raw[:,0,:,(numWindows-1)*windowSize+knifeEdge:numWindows*windowSize-1],65535)
                self.dark = np.tile(self.dark, [3, 1, 1, 1])
                self.dark = np.transpose(self.dark, [1, 0, 2, 3])
                
                for q,p in enumerate(pwaLoc):
                    self.pwa[:,q,:,:] = np.minimum(self.raw[:,0,:,(p-1)*windowSize+knifeEdge:(p)*windowSize-1],65535)
                    
                # Do PCA
                pwoas = np.minimum(self.raw[:,0,:,(numWindows-2)*windowSize+knifeEdge:(numWindows-1)*windowSize-1],65535)
                mean_pwoa = np.mean(pwoas, axis = 0)
                pwoas = pwoas - mean_pwoa
                pwoa_flat = pwoas.reshape( (pwoas.shape[0], np.prod(pwoas.shape[1:])) )
                u, s, vh = np.linalg.svd(pwoa_flat, full_matrices = False)
                
                self.pwoa = np.empty(self.pwa.shape)
                for i in range( self.pwa.shape[0] ):
                    for j in range( self.pwa.shape[1] ):
                        img = self.pwa[i, j] - mean_pwoa
                        projection = np.array([img.ravel() @ vh[j] for j in range(len(vh))])
                        self.pwoa[i, j] = (projection @ vh).reshape( pwoas.shape[1:] ) + mean_pwoa
                
                # Calculate PCI image
                self.pciImg = (self.pwa - self.pwoa) / (self.pwoa - self.dark)
                
                
            else:            
                for q,p in enumerate(pwaLoc):
                    self.pwa[:,q,:,:] = np.minimum(self.raw[:,0,:,(p-1)*windowSize+knifeEdge:(p)*windowSize-1],65535)

                self.pwoa = np.minimum(self.raw[:,0,:,(numWindows-2)*windowSize+knifeEdge:(numWindows-1)*windowSize-1],65535)
                self.dark = np.minimum(self.raw[:,0,:,(numWindows-1)*windowSize+knifeEdge:numWindows*windowSize-1],65535)

                for i in range(len(pwaLoc)):
                    self.pciImg[:,i,:,:] = (self.pwa[:,i,:,:]-self.pwoa)/(self.pwoa-self.dark)
                    self.darkGround[:,i,:,:] = (self.pwa[:,i,:,:]-self.dark)-(self.pwoa-self.dark)*(1+2*(np.sin(0.22-self.pciImg[:,i,:,:]/(-2.34))-np.sin(0.22)))
            
            
            
            return self.pciImg
        except frameNumError:
            print(f"you have {self.frameN} frame(s). You only need one to compute dispersive images")


def pca(pwas, pwoas):
    """
    Construct a PCA basis using the elements in `pwoas`, and apply to `pwas`
    """
    mean_pwoa = np.mean(pwoas, axis = 0)
    pwoas = pwoas - mean_pwoa
    
    pwoa_flat = pwoas.reshape( (pwoas.shape[0], np.prod(pwoas.shape[1:])) )
    u, s, vh = np.linalg.svd(pwoa_flat, full_matrices = False)
    
    recon_pwoas = np.empty(pwas.shape)
    for i in range(len(pwas)):
        img = pwas[i] - mean_pwoa
        projection = np.array([img.ravel() @ vh[j] for j in range(len(vh))])
        recon_pwoas[i] = (projection @ vh).reshape( pwoas.shape[1:] ) + mean_pwoa
    
    return recon_pwoas

def doublonAnalysis(Ncounts, doublon_mode, scan_var):
    """
    Input
        Ncounts: array of ncounts
        doublon_mode: array of 'doublon_mode' variable value
        scan_var: array of specified variable value

    Output
        dblfrac, spdf, dblerr, spdferr
    """

    if type(Ncounts) == list:
        Ncounts = np.array(Ncounts)
    
    unique_vars = np.unique(scan_var)
    nvar = len(unique_vars)
    allAtoms_ave = np.empty(nvar)
    rmDoublons_ave = np.empty(nvar)
    rmPairs_ave = np.empty(nvar)
    
    allAtoms_std = np.empty(nvar)
    rmDoublons_std = np.empty(nvar)
    rmPairs_std = np.empty(nvar)
    
    correction = 1.0
    for i in range(nvar):
        allAtoms = Ncounts[ np.logical_and( doublon_mode == 1, scan_var == unique_vars[i] ) ]
        allAtoms_ave[i] = allAtoms.mean()
        allAtoms_std[i] = allAtoms.std() / np.sqrt( len(allAtoms) - 1 )
        rmPairs = correction**2 * Ncounts[ np.logical_and( doublon_mode == 3, scan_var == unique_vars[i] ) ]
        rmPairs_ave[i] = rmPairs.mean()
        rmPairs_std[i] = rmPairs.std() / np.sqrt( len(rmPairs) - 1 )
        rmDoublons = correction * Ncounts[ np.logical_and( doublon_mode == 2, scan_var == unique_vars[i] ) ]
        rmDoublons_ave[i] = rmDoublons.mean()
        rmDoublons_std[i] = rmDoublons.std() / np.sqrt( len(rmDoublons) - 1 )
        
    dblfrac = (allAtoms_ave - rmDoublons_ave) / allAtoms_ave
    spdf = (allAtoms_ave - rmPairs_ave) / (allAtoms_ave - rmDoublons_ave)
    
    dblerr = dblfrac * np.sqrt( allAtoms_std**2 / allAtoms_ave**2 + rmDoublons_std**2 / rmDoublons_ave**2 )
    spdferr = np.sqrt( (allAtoms_std**2 + rmPairs_std**2) / (allAtoms_ave - rmDoublons_ave)**2 \
                      + (allAtoms_ave - rmPairs_ave)**2 * (allAtoms_std**2 + rmDoublons_std**2) \
                          / (allAtoms_ave - rmDoublons_ave)**4)
        
    return dblfrac, spdf, dblerr, spdferr

def multiFrameAnalysis(Ncounts, frame_number, scan_var):
    """
    Similar to doublonAnalysis, but just returns the three average atom numbers without doing any further operations.
    """
    if type(Ncounts) == list:
        Ncounts = np.array(Ncounts)
    
    unique_vars = np.unique(scan_var)
    nvar = len(unique_vars)
    frame1_ave = np.empty(nvar)
    frame2_ave = np.empty(nvar)
    frame3_ave = np.empty(nvar)
    
    frame1_std = np.empty(nvar)
    frame2_std = np.empty(nvar)
    frame3_std = np.empty(nvar)
    
    correction = 1.0
    for i in range(nvar):
        frame1 = Ncounts[ np.logical_and( frame_number == 1, scan_var == unique_vars[i] ) ]
        frame1_ave[i] = frame1.mean()
        frame1_std[i] = frame1.std() / np.sqrt( len(frame1) - 1 )
        frame2 = correction**2 * Ncounts[ np.logical_and( frame_number == 2, scan_var == unique_vars[i] ) ]
        frame2_ave[i] = frame2.mean()
        frame2_std[i] = frame2.std() / np.sqrt( len(frame2) - 1 )
        frame3 = correction * Ncounts[ np.logical_and( frame_number == 3, scan_var == unique_vars[i] ) ]
        frame3_ave[i] = frame3.mean()
        frame3_std[i] = frame3.std() / np.sqrt( len(frame3) - 1 )
        
    return frame1_ave, frame3_ave, frame2_ave, frame1_std, frame2_std, frame3_std

def spdf_jackknife(ncount,doublon_mode,scan_var):
    """
    Uses jackknifing method to calculate unbiased estimate of SPDF mean and variance

    ncount: array of ncount
    doublon_mode: array of 'doublon_mode' variable value
    scan_var: array of specified variable value

    
    """
    var_unique, unique_key, unique_counts = np.unique(scan_var,return_inverse=True,return_counts=True)
    a = dict() # for  all atoms
    p = dict() # for kill pairs
    d = dict() # for kill doublons
        
    #all_atoms = 

    key_order = np.argsort(unique_key)   
    all_atoms = y_arr[key_order,0] # all atoms
    kill_pairs = y_arr[key_order,1] # kill pairs
    kill_dbl = y_arr[key_order,2] # kill doublons
    
    total = 0
    for val,cts in zip(var_unique,unique_counts):
        a[val] = all_atoms[total:total+cts]
        p[val] = kill_pairs[total:total+cts]
        d[val] = kill_dbl[total:total+cts]
        total += cts
    
    a_mean = {val:a[val].mean() for val in a}
    p_mean = {val:p[val].mean() for val in p}
    d_mean = {val:d[val].mean() for val in d}
    
    
    arr_ind = range(len(var_unique))
    
    dat = np.zeros((len(var_unique),2,2))
    
    for i,val,cts in zip(arr_ind,var_unique,unique_counts):
        a_j = np.fromiter((a_mean[val] + 1/(cts-1)*(a_mean[val] - a_i) for a_i in a[val]),dtype=np.float)
        p_j = np.fromiter((p_mean[val] + 1/(cts-1)*(p_mean[val] - p_i) for p_i in p[val]),dtype=np.float)
        d_j = np.fromiter((d_mean[val] + 1/(cts-1)*(d_mean[val] - d_i) for d_i in d[val]),dtype=np.float)
        
        
        dbl_jbar = np.mean((a_j-d_j)/a_j)
        dbl_jbarSquared = np.mean(np.power((a_j-d_j)/a_j,2))
        
        spdf_jbar = np.mean((a_j-p_j)/(a_j-d_j))
        spdf_jbarSquared = np.mean(np.power((a_j-p_j)/(a_j-d_j),2))
    
        dbl_jackknifed = cts*(a_mean[val]-d_mean[val])/a_mean[val] - (cts-1)*dbl_jbar
        dbl_err_jackknifed = np.sqrt((cts-1)*(dbl_jbarSquared-dbl_jbar**2))
        
        spdf_jackknifed = cts*(a_mean[val]-p_mean[val])/(a_mean[val]-d_mean[val]) - (cts-1)*spdf_jbar
        spdf_err_jackknifed = np.sqrt((cts-1)*(spdf_jbarSquared-spdf_jbar**2))
    
        dat[i,0,0] = dbl_jackknifed
        dat[i,0,1] = dbl_err_jackknifed
        dat[i,1,0] = spdf_jackknifed
        dat[i,1,1] = spdf_err_jackknifed
    
    return dat
    
    
    
    
    