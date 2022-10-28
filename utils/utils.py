import numpy as np
from numpy import arccos, arcsin, cos, pi, power, sin, sqrt


def polar2cartesian(r,phi,theta):
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)

    return x,y,z

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


def makeLookAt(position, target, up):
    
    forward = np.subtract(position, target)
    forward = np.divide(forward, np.linalg.norm(forward))

    right = np.cross(forward, up)
         
    # if forward and up vectors are parallel, right vector is zero; 
    #   fix by perturbing up vector a bit
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array( [0.001, 0, 0] )
        right = np.cross( forward, up + epsilon )
        
    right = np.divide( right, np.linalg.norm(right) )
    
    up = np.cross( right, forward )
    up = np.divide( up, np.linalg.norm(up) )
    
    return np.array([[right[0], up[0], -forward[0], position[0]], 
                        [right[1], up[1], -forward[1], position[1]], 
                        [right[2], up[2], -forward[2], position[2]],
                        [0, 0, 0, 1]])

def nearest_intersection(points, dirs):
        """
        :param points: (N, 3) array of points on the lines
        :param dirs: (N, 3) array of unit direction vectors
        :returns: (3,) array of intersection point
        """
        dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
        points_mat = points[:, :, np.newaxis]
        I = np.eye(3)
        return np.linalg.lstsq(
            (I - dirs_mat).sum(axis=0),
            ((I - dirs_mat) @ points_mat).sum(axis=0),
            rcond=None
            )[0]

def points_on_dome(r, num_points):

    indices = np.arange(0, num_points, dtype=float) + 0.5

    theta = np.arccos(1 - 2*indices/num_points)
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

def points_circle(r, center, num_points):
    return np.array([
            [center[0] + cos(2 * pi / num_points * x) * r, center[1] + sin(2 * pi / num_points * x) * r, center[2]]
                for x in range(0, num_points + 1)
            ])


def n_points_on_vertical_circle(r, center, num_points):

    theta = np.linspace(0,2*np.pi,num_points,endpoint=False); 
    x = r * np.cos(theta)+center[0]
    y = np.broadcast_to(center[1],num_points)
    z =  r * np.sin(theta)+center[2]

    xyz = np.stack((x,y,z),axis=-1)
    return xyz
