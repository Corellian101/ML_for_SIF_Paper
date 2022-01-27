import numpy as np
import matplotlib.pyplot as plt

def F_s(a_c_,a_t_,c_b_,phi_,only_mg=False):
    a_c_ = np.array(a_c_)
    a_t_ = np.array(a_t_)
    c_b_ = np.array(c_b_)
    phi_ = np.array(phi_)

    try:
        F_s = np.zeros(len(a_c_))
        iters = len(a_c_)
    except:
        F_s = np.zeros(1)
        iters = 1
    for i in range(iters):
        a_c = a_c_[i]
        a_t = a_t_[i]
        c_b = c_b_[i]
        phi = phi_[i]
        fw = np.cos((np.pi/2)*(c_b)*np.sqrt(a_t))**(-0.5)
        if np.isnan(a_c):
            return 0
        if a_c <= 1:
            M1 = 1.13 - 0.09*(a_c)
            M2 = -0.54 + (0.89/(0.2+(a_c)))
            M3 = 0.5 - (1/(0.65+(a_c))) + 14*(1-(a_c))**(24)
            g = 1 + (0.1 + 0.35*(a_t)**2)*(1-np.sin(phi))**2
            fphi = ((a_c)**2*(np.cos(phi))**2+(np.sin(phi))**2)**(1/4)
    
        if a_c > 1:
            M1 = np.sqrt((a_c**(-1)))*(1+0.04*((a_c**(-1))))
            M2 = 0.2*((a_c**(-1)))**4
            M3 = -0.11*((a_c**(-1)))**4
            g = 1 + (0.1+0.35*((a_c**(-1)))*(a_t)**2)*(1-(np.sin(phi)))**2
            fphi = ((((a_c**(-1)))**2)*np.sin(phi)**(2)+np.cos(phi)**(2))**(1/4)
        
        if only_mg:
            Fs = (M1+M2*(a_t)**2+M3*(a_t)**4)*g
        else:
            Fs = (M1+M2*(a_t)**2+M3*(a_t)**4)*g*fphi*fw  
        F_s[i] = Fs

    return F_s

def F_s_bingo(a_c,a_t,c_b,phi):
    a_c = np.array(a_c)
    a_t = np.array(a_t)
    c_b = np.array(c_b)
    phi = np.array(phi)
    F_s_bingo = np.zeros(len(a_c))
    for i in range(len(a_c)):
        X_0 = a_c[i]
        X_1 = a_t[i]
        X_2 = c_b[i]
        X_3 = phi[i]
        if X_0 <= 1:
            fphi = ((X_0)**2*(np.cos(X_3))**2+(np.sin(X_3))**2)**(1/4)
            fw = np.cos((np.pi/2)*(X_2)*np.sqrt(X_1))**(-0.5)
            g = np.cos(11071.131423790328) - X_1*(-1.0263496149498648 + np.sin(279.0551708800622 - np.sin(X_3)))
            #g = 0.987 - X_1*(-1.026 + np.sin(279 - np.sin(X_3)))
            M = 1.0052643174132263 - ((X_1)*((X_0 + -5.897146261824969)*(X_0) - ((-8.41374265030078)*(np.sqrt(X_0)) + 3.65713751815375)))
            #M = -X_1*(8.41374265030078*np.sqrt(X_0) + X_0*(X_0 - 5.89714626182497) - 3.65713751815375) + 1.00526431741323
        if X_0 > 1:
            fphi = ((((X_0**(-1)))**2)*np.sin(X_3)**(2)+np.cos(X_3)**(2))**(1/4)
            fw = np.cos((np.pi/2)*(X_2)*np.sqrt(X_1))**(-0.5)
            g = 1 + (0.1+0.35*((X_0**(-1)))*(X_1)**2)*(1-(np.sin(X_3)))**2
            M = 0.5005597384684810*np.sqrt(1/X_0)*np.sqrt(X_1)+0.2965415493436768*(1/X_0)+0.6637021883924064*(1/X_0)**(1/4)-0.3628181028391626*np.sqrt(X_1)+0.02115042312837423
        Fs = M*g*fw*fphi
        F_s_bingo[i] = Fs
    return F_s_bingo

def M(a_c,a_t,phi):
    M1 = []
    M2 = []
    M3 = []      
    for i in range(len(a_c)):
        if a_c <= 1:
            M1.append(1.13 - 0.09*(a_c[i]))
            M2.append(-0.54 + (0.89/(0.2+(a_c[i]))))
            M3.append(0.5 - (1/(0.65+(a_c[i]))) + 14*(1-(a_c[i]))**(24))
    
    
        if a_c > 1:
            M1.append(np.sqrt((a_c[i]**(-1)))*(1+0.04*((a_c[i]**(-1)))))
            M2.append(0.2*((a_c[i]**(-1)))**4)
            M3.append(-0.11*((a_c[i]**(-1)))**4)
    M1 = np.array(M1)
    M2 = np.array(M2)
    M3 = np.array(M3)
    
    return M1, M2, M3

    

def F_ch(a_c,a_t,r_t,r_w,c_w,phi,r,c,w):
    mu = 0.85
    c_r = c_w/r_w
    L = 1/(1+(c_r)*np.cos(mu*phi))
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    for i in range(len(a_c)):
        sec = lambda x: 1/np.cos(x)
        n = 1
        fw = (sec(np.pi/2*r_w)*sec(np.pi*(2*r + n*c)/(r*(w - c) + 2*n*c)*(a_t)**(1/2)))**(1/2)
        if a_c <= 1:
            g1.append(1 + (0.1 + 0.35*(a_t[i])**2)*(1 - np.sin(phi[i])**2))
            g2.append((1 + 0.358*L[i] + 1.425*L[i]**2 - 1.578*L[i]**3 + 2.156*L[i]**4)/(1 + 0.13*L[i]**2))
            g3.append((1 + 0.4*a_c[i])*(1 + 0.1*(1 - np.cos(phi[i]))**2)*(0.85 + 0.15*(a_t[i])**(1/4)))
            g4.append(1 - 0.7*(1 - a_t[i])*(a_c[i] - 0.2)*(1 - a_c[i]))
            fphi = ((a_c[i])**2*(np.cos(phi[i]))**2+(np.sin(phi[i]))**2)**(1/4)
        if a_c > 1:
            g1.append(1 + (0.1 + 0.35*(a_c[i]**(-1))*(a_t[i])**2)*(1 - np.sin(phi[i])**2))
            g2.append((1 + 0.358*L[i] + 1.425*L[i]**2 - 1.578*L[i]**3 + 2.156*L[i]**4)/(1 + 0.13*L[i]**2))
            g3.append((1.13 - 0.09*(a_c[i]**(-1)))*(1 + 0.1*(1 - np.cos(phi[i]))**2)*(0.85 + 0.15*(a_t[i])**(1/4)))
            g4.append(1)
            fphi = ((((a_c**(-1)))**2)*np.sin(phi)**(2)+np.cos(phi)**(2))**(1/4)
    
    M1, M2, M3 = M(a_c,a_t,phi)
    
    F_ch = (M1 + M2*(a_t)**2 + M3*(a_t)**4)*g1*g2*g3*g4*fphi*fw
    
    return F_ch

def F_nf(a_c,a_t,r_t,r_w,c_w,phi,r,c,w):
    f1 = 1 + (2*(a_c)/(1 + (a_c)**2))*(a_t)**20*(2*phi/np.pi)**10
    f2 = []
    f3 = []
    f4 =[]
    for i in range(len(a_c)):
        if a_c <= 1:
            f2.append(1 + 0.75*(1 - 1/(1 + r_t[i]))*((1.2 - a_c[i])*(a_c[i]))**(1.5)*(1 - a_t[i]))
            f3.append(1 - 0.13*(a_c[i])**6*(1-2*phi[i]/np.pi)**4/(1 + 2*r_t[i]))
            f4.append(1 + 0.1*(1 - a_c[i])**(1/2)*(1 - 1/(1 + r_t[i]))*(a_t[i])*(1 - 2*phi[i]/np.pi)**4)
        if a_c > 1:
            c_a = a_c[i]**(-1)
            f2.append(1 + 0.75*(1 - 1/(1 + r_t[i]))*(0.2*(c_a))**(1.5)*(1 - a_t[i]))
            f3.append(1 - 0.13*(1 - 2*phi[i]/np.pi)**4/(1 + 2*r_t[i]*(1 - (1 - (c_a))*(a_t[i]))))
            f4.append(1) 
    f1 = np.array(f1)
    f2 = np.array(f2)
    f3 = np.array(f3)
    f4 = np.array(f4)
    Ftfa = f1*f2*f3
    
    Fch = F_ch(a_c,a_t,r_t,r_w,c_w,phi,r,c,w)
    
    Fnf = Ftfa*Fch
    
    return Fnf
            
    
    
    
    