import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time

class Environment:

	def __init__(self, path):
			
		self.loadNPZ(path)
		
	def loadNPZ(self, path):
	
		npzfile = np.load(path)
		
		self.cloud = npzfile["arr_0"]
		
	def plot(self, axis, ratio = 0):
	
		plot_cloud = self.reduceResolution(ratio) if ratio else self.cloud

		surf = axis.plot_surface(plot_cloud[:, :, 0], plot_cloud[:, :, 1], plot_cloud[:, :, 2], rstride = 1, cstride = 1, cmap = cm.jet)

		return surf
		
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

	start = time.time()

	environment = Environment("cloud.npz")
	
	print( "Time elapsed : {} seconds.".format(time.time() - start) )
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	
	environment.plot(ax, 15)
	
	plt.show()
