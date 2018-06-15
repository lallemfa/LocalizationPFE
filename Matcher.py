import time
import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from Environment import Environment
from Sonar import Sonar

class Matcher:
  
    accepted_method = ["SIFT", "SURF"]
    
    def __init__(self, environment, method = "SURF"):
        self.method = method if method in self.accepted_method else "SURF"
        
        self.init_descriptors(environment)
        
    def __str__(self):
        return "\tMatcher\nMethod : {}\nNumber of keypoints : {}".format(self.method, len(self.original_keypoints))
    
    def plot(self):
        
        key_img = cv2.drawKeypoints(self.rgb_img, self.original_keypoints, None, color=(0,255,0), flags=4)
        
        cv2.imshow("Keypoints on environment with method {}".format(self.method), key_img)
        
        plt.figure("Keypoints on environment with method {}".format(self.method))
        plt.imshow(key_img)
        plt.show()
                
    def init_descriptors(self, environment):
        if self.method == "SIFT":
            self.finder = cv2.xfeatures2d.SIFT_create()
        elif self.method == "SURF":
            self.finder = cv2.xfeatures2d.SURF_create()
            
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	
        [Gx, Gy] = environment.gradient()
    
        self.hsv_img = environment.hsv_gradient(Gx, Gy, environment.cloud[:,:, 2])
        self.rgb_img = cv2.cvtColor(self.hsv_img, cv2.COLOR_HSV2BGR)
        
        [self.original_keypoints, self.original_descriptors] = self.finder.detectAndCompute(self.hsv_img, None)
        
    def match(self, img, number_of_matches = 30):
        
        [keypoints, descriptors] = self.finder.detectAndCompute(img, None)
        
        if len(keypoints) == 0:
            print("No keypoints in this view")
            return None, None, None, None, None
        
        matches = self.bf.match(self.original_descriptors, descriptors)
        
        matches = sorted(matches, key = lambda x:x.distance)
        
        matches = matches[:number_of_matches]
        
        self.matches_plot(img, matches, keypoints)
        
        print("\n")
        for match in matches:
            print(match.distance)
        print("\n")
        
        return matches, self.original_descriptors, self.original_keypoints, descriptors, keypoints
    
    def match_imgs(self, img1, img2, number_of_matches = 5):
        
        [keypoints_img1, descriptors_img1] = self.finder.detectAndCompute(img1, None)
        
        [keypoints_img2, descriptors_img2] = self.finder.detectAndCompute(img2, None)
        
        if len(keypoints_img1) == 0 or len(keypoints_img2) == 0:
            print("No keypoints in one image")
            return None, None, None, None, None
        
        matches = self.bf.match(descriptors_img1, descriptors_img2)
        
        matches = sorted(matches, key = lambda x:x.distance)
        
        matches = matches[:number_of_matches]
        
        self.matches_imgs_plot(matches, img1, img2, keypoints_img1, keypoints_img2)
        
        return matches, descriptors_img1, keypoints_img1, descriptors_img2, keypoints_img2
    
    def matches_imgs_plot(self, matches, img1, img2, keypoints_img1, keypoints_img2):
        
        matching_idx_img1 = [matches[i].queryIdx for i in range(len(matches))]
        matching_idx_img2 = [matches[i].trainIdx for i in range(len(matches))]
        
        kps_img1 = [keypoints_img1[i] for i in range(len(keypoints_img1))]
        kps_img2 = [keypoints_img2[i] for i in range(len(keypoints_img2))]
        
        kps_img1 = [kps_img1[i] for i in matching_idx_img1]
        kps_img2 = [kps_img2[i] for i in matching_idx_img2]
        
        pts_img1 = np.array([keypoints_img1[i].pt for i in range(len(keypoints_img1))])
        pts_img2 = np.array([keypoints_img2[i].pt for i in range(len(keypoints_img2))])
        
        pts_img1 = pts_img1[matching_idx_img1, :]
        pts_img2 = pts_img2[matching_idx_img2, :]
        
# =============================================================================
#         Views of keypoints on images
# =============================================================================
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
        
        kp_img1_view = cv2.drawKeypoints(img1_rgb, kps_img1, None, color=(0,0,255), flags=4)
        cv2.imshow("Kept keypoints on original view", kp_img1_view)
        
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
        
        kp_img2_view = cv2.drawKeypoints(img2_rgb, kps_img2, None, color=(0,0,255), flags=4)
        cv2.imshow("Kept keypoints on new view", kp_img2_view)
        
    def matches_plot(self, img, matches, keypoints):
        
        matching_idx_original_img   = [matches[i].queryIdx for i in range(len(matches))]
        matching_idx_new_img        = [matches[i].trainIdx for i in range(len(matches))]
        
        original_kps    = [self.original_keypoints[i] for i in range(len(self.original_keypoints))]
        new_img_kps     = [keypoints[i] for i in range(len(keypoints))]
        
        original_kps    = [original_kps[i] for i in matching_idx_original_img]
        new_img_kps     = [new_img_kps[i] for i in matching_idx_new_img]
        
        original_pts    = np.array([self.original_keypoints[i].pt for i in range(len(self.original_keypoints))])
        new_img_pts     = np.array([keypoints[i].pt for i in range(len(keypoints))])
        
        original_pts    = original_pts[matching_idx_original_img, :]
        new_img_pts     = new_img_pts[matching_idx_new_img, :]
        
# =============================================================================
#         Views of keypoints on images
# =============================================================================
        
        kp_original_view = cv2.drawKeypoints(self.rgb_img, original_kps, None, color=(0,0,255), flags=4)
        cv2.imshow("Kept keypoints on original view", kp_original_view)
        
        cv2.imwrite("kp_original_view.jpg", kp_original_view)
        
        new_rgb_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        kp_new_view = cv2.drawKeypoints(new_rgb_img, new_img_kps, None, color=(0,0,255), flags=4)
        cv2.imshow("Kept keypoints on new view", kp_new_view)
        
        cv2.imwrite("kp_new_view.jpg", kp_new_view)
        
# =============================================================================
#         Keypoints matching representation via Delaunay triangulation
# =============================================================================

#        if len(matches) >= 3:
#            original_pts[:, 0] = original_pts[:, 0]/original_pts[:, 0].max()
#            original_pts[:, 1] = original_pts[:, 1]/original_pts[:, 1].max()
#            
#            new_img_pts[:, 0] = new_img_pts[:, 0]/new_img_pts[:, 0].max()
#            new_img_pts[:, 1] = new_img_pts[:, 1]/new_img_pts[:, 1].max()
#    
#            original_tri    = Delaunay(original_pts)
#            new_img_tri     = Delaunay(new_img_pts)
#            
#            plt.figure("Strongest matching keypoints comparison via Delaunay triangulation")
#            plt.triplot(original_pts[:,0], original_pts[:,1], original_tri.simplices.copy(), 'g-')
#            plt.triplot(new_img_pts[:,0], new_img_pts[:,1], new_img_tri.simplices.copy(), 'r--')
        
        
if __name__ == '__main__':
    
    plt.close('all')
    cv2.destroyAllWindows()
    
    environment = Environment("cloud.npz")
    
    matcher = Matcher(environment, "SURF")
    
    print(matcher)
    
#    matcher.plot()
    
    [Gx, Gy] = environment.gradient()
    
    ori_hsv_img = environment.hsv_gradient(Gx, Gy, environment.cloud[:, :, 2])
    
# =============================================================================
#     Test partial view
# =============================================================================
    
    
    sonar = Sonar(environment, [600, 600])
    
    T = np.array([450103.16, 4951485.2])
    
#    T = np.array([450500, 4951800])
    
    start = time.time()
    view, _ = sonar.generateView(T, np.pi/4, True)
    print('Time elapsed {}'.format(time.time() - start))
    
    test = view[:,:,2]
    
#    plt.figure("View")
#    plt.imshow(test.T)
#    plt.show()
    
    [Gx, Gy] = environment.gradient(test.T)
    
    new_hsv_img = environment.hsv_gradient(Gx, Gy, test.T)
    
    matches, original_descriptors, original_keypoints, given_img_descriptors, given_img_keypoints = matcher.match(new_hsv_img)
    
#    fig 	= plt.figure("Given HSV")
#    
#    plt.imshow(new_hsv_img)
#    
#    plt.show()
#    
#    fig 	= plt.figure("Original HSV")
#    
#    plt.imshow(ori_hsv_img)
#    
#    plt.show()
    
#    angles = []
#    
#    for match in matches:
#        original_kp     = original_keypoints[match.queryIdx]
#        given_img_kp    = given_img_keypoints[match.trainIdx]
#        
#        angle = original_kp.angle - given_img_kp.angle
#        
#        angles.append(angle)
#        
#    print(angles)
#    
#    angle = np.mean(angles)
#    
#    print(angle)


# =============================================================================
#     Test reduce original image
# =============================================================================

    new_ori_hsv_img = ori_hsv_img[150:350, 150:350, :]
    
    map_rgb = cv2.cvtColor(new_ori_hsv_img, cv2.COLOR_HSV2BGR)
    partial_map_rgb = cv2.cvtColor(new_hsv_img, cv2.COLOR_HSV2BGR)
    
    cv2.imshow("Img1", map_rgb)
    cv2.imshow("Img2", partial_map_rgb)
    

    
    cv2.imwrite("full_map.png", map_rgb)
    cv2.imwrite("partial_map.png", partial_map_rgb)

    matches, original_descriptors, original_keypoints, given_img_descriptors, given_img_keypoints = matcher.match_imgs(new_ori_hsv_img, new_hsv_img)
