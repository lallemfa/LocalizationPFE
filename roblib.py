import numpy as np
import scipy
from math import pi
import matplotlib.pyplot as plt

from numpy.linalg import inv, eig, norm

from scipy.linalg import sqrtm

from matplotlib.patches import Ellipse

def sawtooth(x):
    return (x+pi)%(2*pi)-pi

def rotmat(phi, theta = None, psi = None):
    
    if any(angle is None for angle in [phi, theta, psi]):
        
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]])
        
    else:
    
        cf = np.cos(phi)
        sf = np.sin(phi)
        
        ct = np.cos(theta)
        st = np.sin(theta)
        
        cp = np.cos(psi)
        sp = np.sin(psi)
        
        R = np.array([[ct*cp, -cf*sp+st*cp*sf,  sp*sf+st*cp*cf],
                      [ct*sp,  cp*cf+st*sp*sf, -cp*sf+st*cf*sp],
                      [  -st,           ct*sf,           ct*cf]])
    
    return R

def rotmat2euler(R):
    
    if R.shape == (3, 3):
        phi = theta = psi = 0
        
        phi = np.arctan2(R[2, 1], R[2, 2])
        
        ctheta  = np.sqrt( R[0, 0]**2 + R[1, 0]**2 )
        theta = np.arctan2(-R[2, 0], ctheta)
        
        psi = np.arctan2(R[1, 0], R[0, 0])
        
        return phi, theta, psi
        
    elif R.shape == (2, 2):
        psi = np.arctan2(R[1, 0], R[0, 0])
    
        return psi
    
    else:
        print("Shape of R must be (2, 2) or (3, 3) but was {}".format(R.shape))
        return None

def draw_disk(center, radius, ax, color = 'r'):
    
    e = Ellipse(xy=center, width=2*radius, height=2*radius, angle=0)   
    
    ax.add_artist(e)
    
    e.set_clip_box(ax.bbox)
    e.set_alpha(1.0)
    e.set_facecolor('none')
    e.set_edgecolor(color)

def computeMinorAndMajorAxisAndAngle(Gamma, precision):
    
    A = sqrtm(-2*np.log(1-precision)*Gamma)
        
    eigenvalues, eigenvectors = eig(A)
    
    major_axis = 2*eigenvalues[0]
    minor_axis = 2*eigenvalues[1]
    
    eigenvector_1 = eigenvectors[:, 0]
    
    phi = (np.arctan2(eigenvector_1[1], eigenvector_1[0]))
    angle = phi*180/3.14
    
    return major_axis, minor_axis, angle
    

def draw_ellipse(center, Gamma, precision, ax, color = 'r'):
    
    if (norm(Gamma)==0):
        Gamma=Gamma+0.001*np.eye(len(Gamma[1,:]))
    
    width, height, angle = computeMinorAndMajorAxisAndAngle(Gamma, precision)
    
    e = Ellipse(center, width, height, angle)   
    ax.add_artist(e)
    
    e.set_clip_box(ax.bbox)
    e.set_alpha(1)
    e.set_facecolor('none')
    e.set_edgecolor(color)

#def draw_ellipse_3d(center, Gamma, precision, ax, color = 'r'):
#    
#    if (norm(Gamma)==0):
#        Gamma=Gamma+0.001*np.eye(len(Gamma[1,:]))
#    
#    a, b, phi = computeMinorAndMajorAxisAndAngle(Gamma, precision)
#    
#    b, c, psi = computeMinorAndMajorAxisAndAngle(Gamma, precision)
    
    
    
def covariance_array(array_of_variable):
    number_of_variable = array_of_variable.shape[1]
    
    variables_mean = np.mean(array_of_variable, axis = 0)
    
    variables_tilde = array_of_variable - np.matlib.repmat(variables_mean, array_of_variable.shape[0], 1)
    
    covariance_array = np.eye(number_of_variable)
    
    for i in range(number_of_variable):
        current_var_tilde = variables_tilde[:, i]
        covariance_array[i, i] = np.mean(current_var_tilde*current_var_tilde)
        
        for j in range(i, number_of_variable):
            other_var_tilde = variables_tilde[:, j]
            value = np.mean(current_var_tilde*other_var_tilde)
            
            covariance_array[i, j] = value
            covariance_array[j, i] = value
    
    return covariance_array
    

def kalman_predict(x, A, Gamma_x, u, B, Gamma_evolution_noise):
    
    Gamma_x = A @ Gamma_x @ A.T + Gamma_evolution_noise
    
    x = A @ x + B @ u    
    
    return(x, Gamma_x)    

def kalman_correc(x, Gamma_x, y, Gamma_observation, C):
    
    S = C @ Gamma_x @ C.T + Gamma_observation        
    
    K = Gamma_x @ C.T @ inv(S)           
    
    ytilde = y - C @ x
    
    Gamma_x = (np.eye(len(x))-K @ C) @ Gamma_x 
    
    x = x + K@ytilde
    
    return(x, Gamma_x) 
    
def kalman(x, A, Gamma_x, u, B, Gamma_evolution_noise, y = None, Gamma_observation = None, C = None):
    
    if not y is None and not Gamma_observation is None and not C is None:
        x, Gamma_x = kalman_correc(x, Gamma_x, y, Gamma_observation, C)
    
    x, Gamma_x = kalman_predict(x, A, Gamma_x, u, B, Gamma_evolution_noise)
    
    return(x, Gamma_x)     

if __name__ == '__main__':
    
    plt.close("all")
    
# =============================================================================
#     Example with arbitraty Gamma and center
# =============================================================================
    
    center = np.array([0, 0])
    
    Gamma1 = np.array([[2.0, 0.0],
                       [0.0, 1.0]])
    
    Gamma2 = np.array([[2.0, 1.0],
                       [1.0, 1.0]])
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, aspect='equal')
    draw_ellipse(center, Gamma1, 0.9, ax, 'r')
    draw_ellipse(center, Gamma2, 0.9, ax, 'g')

    plt.axis([-25, 25, -25, 25])
    plt.show()
    
    
# =============================================================================
#     Example with normal distribution
# =============================================================================
    
    x = np.random.randn(10000, 1)
    y = np.random.randn(10000, 1)
    
    z = x + 5*y
    
    xtilde = x - np.mean(x)
    ytilde = y - np.mean(y)
    ztilde = z - np.mean(z)
    
    
    center_xz = np.array([np.mean(x), np.mean(z)])
    center_yz = np.array([np.mean(y), np.mean(z)])
    
    Gxz = covariance_array(np.hstack((x, z)))
    Gyz = covariance_array(np.hstack((y, z)))
    
    
    fig = plt.figure()
    
    ax = fig.add_subplot(121, aspect='equal')
    draw_ellipse(center_xz, Gxz, 0.9, ax, 'r')
    draw_ellipse(center_xz, Gxz, 0.99, ax, 'g')
    ax.plot(x, z, '.', zorder = 1)

    
    ax = fig.add_subplot(122, aspect='equal')
    draw_ellipse(center_yz, Gyz, 0.9, ax, 'r')
    draw_ellipse(center_yz, Gyz, 0.99, ax, 'g')
    ax.plot(y, z, '.', zorder = 1)
    
    plt.show()    

# =============================================================================
#     Example 3D
# =============================================================================
    
#    center = np.array([0, 0, 0])
#    
#    Gamma1 = np.array([[2.0, 0.0, 0.0],
#                       [0.0, 1.0, 0.0],
#                       [0.0, 0.0, 1.8]])
#    
#    fig = plt.figure()
#    
#    ax = fig.add_subplot(111, projection = '3d')
#    draw_ellipse_3d(center, Gamma1, 0.9, ax, 'r')
#
#    plt.axis([-25, 25, -25, 25])
#    plt.show()
    
# =============================================================================
#     Conversion angles / matrice
# =============================================================================

    phi     = 1.4
    theta   = 0.8
    psi     = 2.4
    
    R = rotmat(phi, theta, psi)
    
    print(R)
    
    a, b, c = rotmat2euler(R)
    
    print(a, b, c)
    
    psi = 0.6
    
    R = rotmat(psi)
    print(R)
    
    a = rotmat2euler(R)
    print(a)
    
    
    
    
    