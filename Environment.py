import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import cv2

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
		
    def plot(self, axis, ratio = 0):
	
        plot_cloud = self.reduceResolution(ratio) if ratio else self.cloud

        surf = axis.plot_surface(plot_cloud[:, :, 0], plot_cloud[:, :, 1], plot_cloud[:, :, 2], rstride = 1, cstride = 1, cmap = cm.jet)

        return surf
    
    def plot_contours(self, axis, n_contours):

        surf = axis.contour(self.cloud[:, :, 0], self.cloud[:, :, 1], self.cloud[:, :, 2], n_contours)

        return surf
    
    def gradient(self, surface = [], ratio = 1):
        
        if not len(surface) :
            cloud = self.reduceResolution(ratio) if ratio else self.cloud
            cloud = cloud[:, :, 2]
        else:
            cloud = surface

        [Gx, Gy] = np.gradient(cloud)
        
        return [Gx, Gy]
    
    def plot_gradient(self, ratio):
        
        [Gx, Gy] = self.gradient(ratio = ratio)
        
        surf = plt.quiver(Gx, Gy)

        return surf
    
    def hsv_gradient(self, Gx, Gy, Z = None, angle = 0):
        
        
        
        hsv_grad = np.zeros(shape=(Gx.shape[0], Gx.shape[1], 3))
        
#        print(len(Gx), len(Gy))
        
        for i in range(len(Gx)):
            for j in range(len(Gx[0])):
                
#                correct_base_Gx = np.cos(angle)*Gx[i, j] - np.sin(angle)*Gy[i, j]
#                correct_base_Gy = np.sin(angle)*Gx[i, j] + np.cos(angle)*Gy[i, j]
                
                hsv_grad[i, j, 0] = (np.arctan2(Gx[i, j], Gy[i, j]) + np.pi) * (180/np.pi) / 2
#                hsv_grad[i, j, 0] = (np.arctan2(correct_base_Gx, correct_base_Gy) + np.pi) * (180/np.pi) / 2
                
                hsv_grad[i, j, 1] = np.sqrt( Gx[i, j]**2 + Gy[i, j]**2 ) * 255
                hsv_grad[i, j, 2] = 255
                hsv_grad[i, j, 2] = 255-abs(Z[i, j]) if not Z is None else 0

        return hsv_grad.astype(np.uint8)
		
    def reduceResolution(self, ratio):
    		
        factor = max(1, ratio)
        		
        shape = self.cloud.shape
        		
        Nrows 		= shape[0]
        Ncolumns 	= shape[1]
        		
        idxX = [i for i in range(0, Nrows, factor)]
        idxY = [i for i in range(0, Ncolumns, factor)]
        
        reduced_cloud = self.cloud[idxX, :, :]
        reduced_cloud = reduced_cloud[:, idxY, :]
        
        return reduced_cloud
    
    def limitsOfEnvironment(self):
        return [self.cloud[:, :, 0].min(), self.cloud[:, :, 0].max(), self.cloud[:, :, 1].min(), self.cloud[:, :, 1].max()]



if __name__ == '__main__':
    
    plt.close("all")
    
    start = time.time()
    
    environment = Environment("cloud.npz")
#    environment = Environment()
    
    print( "Time elapsed : {} seconds.".format(time.time() - start) )
    
# =============================================================================
#     Surface plot
# =============================================================================
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    environment.plot(ax, 15)
    
    plt.show()
    
# =============================================================================
#     Contours of the floor
# =============================================================================
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    environment.plot_contours(ax, 50)
    
    plt.show()
    
# =============================================================================
#     Gradient quiver representation
# =============================================================================
    
    fig = plt.figure()
    
    environment.plot_gradient(2)
    
    plt.show()
    
# =============================================================================
#     Gradient HSV representation
# =============================================================================
    
    [Gx, Gy]    = environment.gradient()
    
    hsv_img     = environment.hsv_gradient(Gx, Gy)
    
    rgb_img     = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    cv2.imshow("Image", rgb_img)


    fig 	= plt.figure("Original HSV")
    
    plt.imshow(hsv_img)
    
    plt.show()






