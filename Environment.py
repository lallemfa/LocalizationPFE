import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import cv2
from scipy.interpolate import griddata

def rotmat(psi):
    
    R = np.array([[np.cos(psi), -np.sin(psi)],
                  [np.sin(psi),  np.cos(psi)]])
    
    return R

class Environment:

    def __init__(self, path = None):
        if path is None:
            X, Y = np.mgrid[-25:26,-25:26]
            Z = np.cos(X/12)*np.sin(Y/12)
            
            self.cloud = np.dstack((X, Y, Z))
        else:
            self.loadNPZ(path)
    		
    def loadNPZ(self, path):
    	
        npzfile = np.load(path)
        		
        self.cloud = npzfile["arr_0"]
    
    def limitsOfEnvironment(self):
        return [self.cloud[:, :, 0].min(), self.cloud[:, :, 0].max(), self.cloud[:, :, 1].min(), self.cloud[:, :, 1].max()]

    def reduceResolution(self, precision, method = "cubic"):
        
        minX = np.nanmin(self.cloud[:, :, 0])
        minY = np.nanmin(self.cloud[:, :, 1])
        
        maxX = np.nanmax(self.cloud[:, :, 0])
        maxY = np.nanmax(self.cloud[:, :, 1])
        		
        gridX, gridY = np.mgrid[minX:maxX+1:precision, minY:maxY+1:precision]
        
        pts = np.reshape(self.cloud, (1, -1, 3))[0, :, :]
        
        pts = pts[~np.isnan(pts[:, 2])]
        
        reduced_cloud = griddata(pts[:, 0:2], pts[:, 2], (gridX, gridY), method = method)
        
        reduced_cloud = np.dstack((gridX, gridY, reduced_cloud.T))
        
        return reduced_cloud
        
# =============================================================================
# =============================================================================
# #     PLOT FUNCTIONS
# =============================================================================
# =============================================================================
		
    def plot(self, axis, ratio = 0):
	
        cloud = self.reduceResolution(ratio) if ratio else self.cloud

        surf = axis.plot_surface(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2], rstride = 1, cstride = 1, cmap = cm.jet)
#        surf = axis.plot_surface(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2], cmap = cm.jet)
        
        return surf
    
    def plot_gradient(self, ratio = 0):
        
        cloud = self.reduceResolution(ratio) if ratio else self.cloud
        cloud = cloud[:, :, 2]
        
        [Gx, Gy] = np.gradient(cloud)
        
        surf = plt.quiver(Gx, Gy)

        return surf
    
    def plot_contours(self, axis, n_contours):

        surf = axis.contour(self.cloud[:, :, 0], self.cloud[:, :, 1], self.cloud[:, :, 2], n_contours)

        return surf
    
if __name__ == '__main__':
    
    plt.close("all")
    
    start = time.time()
    
    environment = Environment("Environment_data/area_2/cloud.npz")
#    environment = Environment()
    
    print( "Time elapsed : {} seconds.".format(time.time() - start) )
    
# =============================================================================
#     Surface plot
# =============================================================================
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
#    
#    environment.plot(ax, 3)
#    
#    plt.show()
    
    environment.reduceResolution(2)

# =============================================================================
#     Contours of the floor
# =============================================================================
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
#    
#    environment.plot_contours(ax, 25)
#    
#    plt.show()

# =============================================================================
#     Gradient quiver representation
# =============================================================================
    
#    fig = plt.figure()
#    
#    environment.plot_gradient(2)
#    
#    plt.show()
    
    
    
# =============================================================================
#     Gradient HSV representation
# =============================================================================
    
#    start = time.time()
#    
#    hsv_img     = environment.hsv_gradient(20)
#    
#    print( "Time elapsed : {} seconds.".format(time.time() - start) )
#
#
#    fig 	= plt.figure("Original HSV")
#    
#    plt.imshow(hsv_img)
#    
#    plt.show()

    
