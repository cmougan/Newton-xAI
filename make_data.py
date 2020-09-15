import numpy as np
import pandas as pd
from scipy.constants import G

def newton_equation(m1=1,m2=1,r=1,G=G):
    '''
    Newton´s Equation
    :param m1: Mass first body in kg
    :param m2: Mass second body in kg
    :param r: Distance between m1 and m2 in meters
    :param G: Gravitational constant
    :return: Newtons equation in the Internation System of Units
    '''
    return G * m1 * m2 * (1/r**2)

def make_newton(samples=100,m1_min=0.001,m1_max=1,m2_min=0.001,m2_max=1,r_min=1,r_max=10,G=G,dataframe=True):
    '''
    Creates a sample of data of the Newton Equation
    :param samples: Number of samples
    :param m1_min: Minimum value in the range of the mass of the first body
    :param m1_max: Maximum value in the range of the mass of the first body
    :param m2_min: Minimum value in the range of the mass of the second body
    :param m2_max: Maximum value in the range of the mass of the second body
    :param r_min: Minimum value of the distance between the bodies
    :param r_max: Maximum value of the distance between the bodies
    :param dataframe: wether it returns a dataframe or an array
    :return: data sample
    '''
    data = []
    for n in range(samples):
        m1 = (m1_max - m1_min) * np.random.random() + m1_min
        m2 = (m2_max - m2_min) * np.random.random() + m2_min
        r = (r_max - r_min) * np.random.random() + r_min

        data.append([m1,
                     m2,
                     r,
                     newton_equation(m1=m1,m2=m2,r=r,G=G)
                     ])
    if dataframe:
        return pd.DataFrame(data=data,columns=['m1','m2','r','f'])
    else:
        return data
