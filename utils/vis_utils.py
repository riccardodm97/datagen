import numpy as np 
import matplotlib.pyplot as plt 
from numpy import pi, cos, sin, arccos, arcsin, sqrt, power, arange

def polar2cartesian(r,phi,theta):
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)

    return x,y,z

def a(r, u, v, m=1):
    theta = arccos(power(1 - u, 1 / (1 + m)))
    phi = 2 * pi * v

    return polar2cartesian(r,phi,theta)

def b(r,u,v):
    phi = 2 * pi * u
    theta = arccos(1.0 - v)

    return polar2cartesian(r,phi,theta)

def c(r,u,v):
    phi = 2 * pi * u
    theta = arcsin(sqrt(v)) 

    return polar2cartesian(r,phi,theta)

def d(r,u,v):

    phi = 2 * pi * u
    theta = pi/2 * v 

    return polar2cartesian(r,phi,theta)

def fibonacci_sphere(samples):

    xs,ys,zs = [], [], []
    phi = pi * (3. - sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = cos(theta) * radius
        z = sin(theta) * radius

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return xs,ys,zs


def samples_uniformly_hemisphere():

    u = np.random.uniform(low=0.0,high=1.0,size=100)
    v = np.random.uniform(low=0.0,high=1.0,size=100)

    radius_c = 0.8
    poi = np.array([0.8,0.2,0])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    x_c,y_c,z_c = b(radius_c,u,v)
  
    ax.scatter(x_c,y_c,z_c,c='blue')
    ax.scatter(*poi, c='green')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def points_on_dome(r,samples):

    indices = np.arange(0, samples, dtype=float) + 0.5

    theta = np.arccos(1 - 2*indices/samples)
    phi = np.pi * (1 + sqrt(5)) * indices

    x,y,z = polar2cartesian(r,phi,theta)

    xyz = np.stack((x,y,z),axis=-1)
    xyz = xyz[xyz[:,-1] > 0]  #take only points above poi 

    return xyz
    
def n_points_on_circle(r, center, num_points):

    theta = np.linspace(0,2*np.pi,num_points,endpoint=False); 
    x = r * np.cos(theta)+center[0]
    y = r * np.sin(theta)+center[1]
    z = np.broadcast_to(center[2],num_points)

    xyz = np.stack((x,y,z),axis=-1)
    return xyz




 
def poses_uniformly_on_dome_and_circle_light():


    radius_c = 0.65
    poi = np.array([0.8,0.2,0.2])


    xyz_c = points_on_dome(radius_c,200)
    xyz_c = xyz_c + poi                       #DEBUG

    theta = np.radians(70)
    circle1_center = poi.copy()
    circle1_center[2]+= radius_c * np.cos(theta)                #DEBUG = oppure += ? la dome è centrata in 0 ma poi tutti i punti vengono alzati di poi[2]
    circle1_radius = radius_c * np.sin(theta) + 0.4
    xyz_l1 = n_points_on_circle(circle1_radius,circle1_center,6)

    # theta = np.radians(40)
    # circle2_center = poi.copy()
    # circle2_center[2]+= radius_c * np.cos(theta)                #DEBUG = oppure += ? la dome è centrata in 0 ma poi tutti i punti vengono alzati di poi[2]
    # circle2_radius = radius_c * np.sin(theta) + 0.4
    # xyz_l2 = n_points_on_circle(circle2_radius,circle2_center,6)

    

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_c[:,0],xyz_c[:,1],xyz_c[:,2],c='blue')
    ax.scatter(xyz_l1[:,0],xyz_l1[:,1],xyz_l1[:,2],c='red')
    # ax.scatter(xyz_l2[:,0],xyz_l2[:,1],xyz_l2[:,2],c='green')
    ax.scatter(*poi, c='black')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

poses_uniformly_on_dome_and_circle_light()

def two_domes():


    radius_c = 0.65
    radius_l = radius_c + 0.2
    poi = np.array([0.8,0.2,0.2])


    xyz_c = points_on_dome(radius_c,200)
    xyz_c = xyz_c + poi                       #DEBUG

    xyz_l = points_on_dome(radius_l,200)
    xyz_l = xyz_l + poi  


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_c[:,0],xyz_c[:,1],xyz_c[:,2],c='blue')
    ax.scatter(xyz_l[:,0],xyz_l[:,1],xyz_l[:,2],c='red')
    ax.scatter(*poi, c='green')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


# two_domes()



