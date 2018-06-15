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
        
        kp1, des1 = self.finder.detectAndCompute(img1,None)
        kp2, des2 = self.finder.detectAndCompute(img2,None)
        
        if des2 is None or des2.shape[0] == 1:
            print("Not enough keypoints are found on base image")
            return center, R, T
        
        matches = self.flann.knnMatch(des1, des2, k = 2)
        
        # store all the good matches as per Lowe's ratio test.
        
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
                
                x = int(round(np.mean(rigid_pts[:, 0, 0])))
                
                y = int(round(np.mean(rigid_pts[:, 0, 1])))
                
                center = (x, y)
                
                false_center = np.array([[50],[50]])
                false_center = R@false_center + T
                
                img2 = cv2.line(img2, (0, 0), (int(T[0,0]), int(T[1,0])), (255,0,0), 3)
                img2 = cv2.line(img2, (0, 0), (int(false_center[0,0]), int(false_center[1,0])), (0,255,0), 3)
                img2 = cv2.polylines(img2, [np.int32(rigid_pts)], True, (0, 255, 0), 3, cv2.LINE_AA)
                img2 = cv2.circle(img2, center, 3, (255,0,0))
                
                if plot_flag:
                    self.plotMatch(img1, img2, kp1, kp2, good)
            
        else:
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
    
    
#    img2 = cv2.imread('full_map_hsv.png') # trainImage
#    img1 = cv2.imread("full_map_hsv.png") # queryImage
#    
#    sol = matcher.matchImages(img1, img2)
    
    centers         = None
    rotations       = None
    translations    = None
    
    
    
    
    for i in range(1,316):
#        img2 = cv2.imread('full_map_hsv.png') # trainImage
        img2 = cv2.imread("Images/HQ_sonar_img_{}.png".format(i-1)) # queryImage
#        img2 = cv2.imread("Images/map_img_{}.png".format(i)) # trainImage
        
        img1 = cv2.imread("Images/HQ_sonar_img_{}.png".format(i)) # queryImage
        
        center, R, T = matcher.matchImages(img1, img2, False)
#        center, R, T = matcher.matchImages(img1, img2)
        
        if not center is None:
            centers         = np.dstack((centers, center)) if not centers is None else center
            rotations       = np.dstack((rotations, R)) if not rotations is None else R
            translations    = np.dstack((translations, T)) if not translations is None else T
        
        
        
        
        
        
        
    angles = [0]
    state = np.array([[0.0],
                      [0.0]])
    
    diffs = np.array([[0.0], [0.0]])
        
    for i in range(1, centers.shape[2]):
        
        previous_psi    = angles[-1]
        previous_X      = state[:, -1]
        
# =============================================================================
#         Rotation
# =============================================================================
                
        R = rotations[:, :, i]
        
        psi = rotmat2euler(R)
        
        angles.append(previous_psi + psi)
#        angles.append(psi)
        
# =============================================================================
#         Translation
# =============================================================================
        
        trans = translations[:, :, i]
        
#        X = np.array([[50],[50]])
#        X = R@X + trans

        
        
        X = previous_X + rotmat(-previous_psi)@trans

#        R2 = rotmat(-previous_psi)

#        diff_center = centers[:, :, i].T - centers[:, :, i-1].T
        
#        print(centers[:, :, i].T)
#        print(centers[:, :, i-1].T)
#        print(diff_center)
        
        
#        diffs = np.hstack((diffs, diff_center))
        
#        
        
#        print(previous_X)
        
#        center = np.array([[50],[50]])
        
#        X = np.reshape(state[:, -1], (2, 1)) + R2@trans
#        X = R2@center + trans
#        X = diffs[:, -1] + previous_X
        
#        X = np.reshape(X, (2, 1))
        
#        print(X)
#        print(X.shape)
#        print(state.shape)
#        print("\n")
        
        
        
        state = np.hstack((state, X))
    
    plt.figure("centers")
    plt.plot(centers[0, 0, :], centers[0, 1, :], 'b.')
    plt.show()
    
    plt.figure("Estimate")
    plt.plot(state[0, :15], state[1, :15], 'b.')
    plt.show()
    
    plt.figure("Angles")
    plt.plot(angles, 'b.')
    plt.show()
    
    plt.figure("Translations")
    plt.plot(translations[0,0,:], translations[1,0,:], 'b.')
    plt.show()
    