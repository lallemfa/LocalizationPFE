import numpy as np
import cv2
from matplotlib import pyplot as plt

from roblib import rotmat2euler, rotmat

class StructMatch:
    
    accepted_method = ["SIFT", "SURF"]
    MIN_MATCH_COUNT = 5
    
    def __init__(self, method = "SURF"):
        
        method = method if method in self.accepted_method else "SURF"
        
# =============================================================================
#         Finder
# =============================================================================
        
        if method == "SIFT":
            self.finder = cv2.xfeatures2d.SIFT_create()
        elif method == "SURF":
            self.finder = cv2.xfeatures2d.SURF_create()
            
# =============================================================================
#         Flann Matcher
# =============================================================================
            
        FLANN_INDEX_KDTREE = 0
        
        index_params    = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params   = dict(checks = 50)
        
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        
    def matchImages(self, img1, img2, plot_flag = True):
        
# =============================================================================
#         Default return
# =============================================================================
        
        center = None
        R = None
        T = None
        
# =============================================================================
#         Cmnpute descriptors amd matches
# =============================================================================
        
        kp1, des1 = self.finder.detectAndCompute(img1, None)
        kp2, des2 = self.finder.detectAndCompute(img2, None)
        
        if (des2 is None or des2.shape[0] == 1 or des1 is None or des1.shape[0] == 1):
            if __name__ == '__main__':
                print("Not enough keypoints are found on base image")
            return center, R, T
        
        matches = self.flann.knnMatch(des1, des2, k = 2)
        
# =============================================================================
#         Keep good matches according to Lowe's ratio test
# =============================================================================
        
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
                
# =============================================================================
#         Compute transformation if enough matches
# =============================================================================
                
        if (len(good) > self.MIN_MATCH_COUNT):
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            sol = cv2.estimateRigidTransform(src_pts, dst_pts, False)

            if not sol is None:
                
                R = sol[0:2, 0:2]
                
                T = np.reshape(sol[:, 2], (2,1))
                
                shape = img1.shape
                
                heigth = shape[0]
                width = shape[1]
                
                pts = np.float32([[      0,        0],
                                  [      0, heigth-1],
                                  [width-1, heigth-1],
                                  [width-1,        0]]).reshape(-1,1,2)
                
                rigid_pts = R@(pts[:,0,:]).T + T
    
                rigid_pts = np.reshape(rigid_pts.T, (4, 1, 2))
                
                x = np.mean(rigid_pts[:, 0, 0])
                
                y = np.mean(rigid_pts[:, 0, 1])
                
                center = (x, y)
                
                false_center = np.array([[50],[50]])
                false_center = R@false_center + T
                

                
                if plot_flag:
                    img2 = cv2.line(img2, (0, 0), (int(T[0,0]), int(T[1,0])), (255,0,0), 3)
                    img2 = cv2.line(img2, (0, 0), (int(false_center[0,0]), int(false_center[1,0])), (0,255,0), 3)
                    img2 = cv2.polylines(img2, [np.int32(rigid_pts)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    self.plotMatch(img1, img2, kp1, kp2, good)
            
        elif __name__ == '__main__':
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            
        return center, R, T
    
        
    def plotMatch(self, img1, img2, kp1, kp2, good):
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
    
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        cv2.imshow("Match", img3)
        cv2.waitKey(0)
        
            
if __name__ == '__main__':
    
    matcher = StructMatch("SURF")
    
    centers         = None
    rotations       = None
    translations    = None
    
    for i in range(1,15902,50):
        img2 = cv2.imread('Environment_data/area_3/Images/RGB/map_rgb_2m.png')
#        img2 = cv2.imread("Environment_data/area_3/Images/RGB/RGB_sonar_view_state{}.png".format(i-50))
        
        
        img1 = cv2.imread("Environment_data/area_3/Images/RGB/RGB_sonar_view_state{}.png".format(i))     
        
        
        center, R, T = matcher.matchImages(img1, img2, False)
#        center, R, T = matcher.matchImages(img1, img2)
        
        if not center is None:
            centers         = np.dstack((centers, center)) if not centers is None else center
            rotations       = np.dstack((rotations, R)) if not rotations is None else R
            translations    = np.dstack((translations, T)) if not translations is None else T
        

 
    plt.figure("centers")
    plt.plot(centers[0, 0, :], centers[0, 1, :], 'b.')
    plt.show()

