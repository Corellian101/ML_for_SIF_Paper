import numpy as np
import math as m

def F_s(a_c,a_t,c_b,phi):
    fw = m.cos((m.pi/2.)*(c_b)*m.sqrt(a_t))**(-0.5)
    if m.isnan(a_c):
        return 0
    if a_c <= 1.:
        M1 = 1.13 - 0.09*(a_c)
        M2 = -0.54 + (0.89/(0.2+(a_c)))
        M3 = 0.5 - (1./(0.65+(a_c))) + 14.*(1.-(a_c))**(24.)
        g = 1. + (0.1 + 0.35*(a_t)**2.)*(1-np.sin(phi))**2.
        fphi = ((a_c)**2.*(np.cos(phi))**2.+(np.sin(phi))**2.)**(1./4.)

    if a_c > 1.:
        M1 = np.sqrt((a_c**(-1.)))*(1.+0.04*((a_c**(-1.))))
        M2 = 0.2*((a_c**(-1.)))**4.
        M3 = -0.11*((a_c**(-1.)))**4.
        g = 1. + (0.1+0.35*((a_c**(-1.)))*(a_t)**2.)*(1.-(np.sin(phi)))**2.
        fphi = ((((a_c**(-1.)))**2.)*np.sin(phi)**(2.)+np.cos(phi)**(2.))**(1./4.)

    Fs = (M1+M2*(a_t)**2.+M3*(a_t)**4.)*g*fphi*fw  

    return Fs

