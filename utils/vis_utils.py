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


def golden_spiral(radius,samples):

    indices = arange(0, samples, dtype=float) + 0.5

    theta = arccos(1 - 2*indices/samples)
    phi = pi * (1 + 5**0.5) * indices

    return polar2cartesian(radius,phi,theta)


 
def samples_uniformly_hemisphere():

    u = np.random.uniform(low=0.0,high=1.0,size=100)
    v = np.random.uniform(low=0.0,high=1.0,size=100)

    radius_c = 0.8
    radius_l = radius_c + 0.5
    poi = np.array([0.8,0.2,0])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    x_c,y_c,z_c = golden_spiral(radius_c,100)
    #x_l,y_l,z_l = golden_spiral(radius_l,100)


    xyz = np.stack((x_c,y_c,z_c),axis=-1)

    x_c+=poi[0]
    y_c+=poi[1]

    # x_l+=poi[0]
    # y_l+=poi[1]


    ax.scatter(x_c,y_c,z_c,c='blue')
    #ax.scatter(x_l,y_l,z_l,c='red')
    ax.scatter(*poi, c='green')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


samples_uniformly_hemisphere()



