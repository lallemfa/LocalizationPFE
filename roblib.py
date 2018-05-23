import numpy as np
import matplotlib.pyplot as plt
from numpy import mean,pi,cos,sin,sqrt,tan,arctan,arctan2,tanh,arcsin,\
                    exp,dot,array,log,inf, eye, zeros, ones, inf,size,\
                    arange,reshape,concatenate,vstack,hstack,diag,median,sign,sum,meshgrid,cross,linspace,append,round
from matplotlib.pyplot import *
from numpy.random import randn,rand
from numpy.linalg import inv, det, norm, eig
from scipy.linalg import sqrtm,expm,norm,block_diag
from scipy.signal import place_poles
from mpl_toolkits.mplot3d import Axes3D
from math import factorial

from matplotlib.patches import Ellipse,Rectangle,Circle, Wedge, Polygon, Arc

from matplotlib.collections import PatchCollection



def eulermat(phi,theta,psi):
    Ad_i = array([[0, 0, 0],[0,0,-1],[0,1,0]])
    Ad_j = array([[0,0,1],[0,0,0],[-1,0,0]])
    Ad_k = array([[0,-1,0],[1,0,0],[0,0,0]])
    M = expm(psi*Ad_k) @ expm(theta*Ad_j) @ expm(phi*Ad_i)
    return(M)

    
def move_motif(M,x,y,theta):
    M1=ones((1,len(M[1,:])))
    M2=vstack((M, M1))
    R = array([[cos(theta),-sin(theta),x], [sin(theta),cos(theta),y]])
    return(R @ M2)    

def translate_motif(R,x):
    return   R + x @ ones((1,R.shape[1]))

def motif_auv3D(): #needed by draw_auv3d and sphere
    return array([ [0.0,0.0,10.0,0.0,0.0,10.0,0.0,0.0],
                   [-1.0,1.0,0.0,-1.0,-0.2,0.0,0.2,1.0],
                   [0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0]])
    
def draw_auv3D(ax,x,phi,theta,psi,col='blue',size=1):   
    M=size*eulermat(phi,theta,psi) @ motif_auv3D()
    M=translate_motif(M,x[0:3].reshape(3,1)) 
    ax.plot(M[0],M[1],1*M[2],color=col)
    ax.plot(M[0],M[1],0*M[2],color='grey')
    
def draw_arrow3D(ax,x,w,col):  # initial point : x ; final point x+w 
    x,w=x.flatten(),w.flatten()  
    ax.quiver(x[0],x[1],x[2],w[0],w[1],w[2],color=col,lw=1,pivot='tail',length=1)
  
def draw_ellipse(c,Γ,η,ax,col): # Gaussian confidence ellipse with artist
    #draw_ellipse(array([[1],[2]]),eye(2),0.9,ax,[1,0.8-0.3*i,0.8-0.3*i])
    if (norm(Γ)==0):
        Γ=Γ+0.001*eye(len(Γ[1,:]))
    A=sqrtm(-2*log(1-η)*Γ)    
    w, v = eig(A)    
    v1=array([[v[0,0]],[v[1,0]]])
    v2=array([[v[0,1]],[v[1,1]]])        
    f1=A @ v1
    f2=A @ v2      
    phi =  (arctan2(v1 [1,0],v1[0,0]))
    α=phi*180/3.14
    e = Ellipse(xy=c, width=2*norm(f1), height=2*norm(f2), angle=α)   
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.7)
    e.set_facecolor(col)
    

def kalman_predict(xup,Gup,u,Γα,A):
    Γ1 = A @ Gup @ A.T + Γα
    x1 = A @ xup + u    
    return(x1,Γ1)    

def kalman_correc(x0,Γ0,y,Γβ,C):
    S = C @ Γ0 @ C.T + Γβ        
    K = Γ0 @ C.T @ inv(S)           
    ytilde = y - C @ x0        
    Gup = (eye(len(x0))-K @ C) @ Γ0 
    xup = x0 + K@ytilde
    return(xup,Gup) 
    
def kalman(x0,Γ0,u,y,Γα,Γβ,A,C):
    xup,Gup = kalman_correc(x0,Γ0,y,Γβ,C)
    x1,Γ1=kalman_predict(xup,Gup,u,Γα,A)
    return(x1,Γ1)     

    
def sawtooth(x):
    return (x+pi)%(2*pi)-pi   # or equivalently   2*arctan(tan(x/2))

