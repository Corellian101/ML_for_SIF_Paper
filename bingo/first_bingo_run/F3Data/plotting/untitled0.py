import numpy as np
from sympy import Symbol, integrate, init_printing, diff, simplify, Matrix, print_latex, sqrt, cos, sin, expand, pi, Eq, solve
#%%
L = Symbol('L')
E = Symbol('E')
I = Symbol('I')
x = Symbol('x')
w = Symbol('w')


#%% Problem 2

R = Symbol('R')
P = Symbol('P')
L = Symbol('L')
theta = Symbol('theta')

a = (R*sin(theta)*P - (R+L)*P)*(R*sin(theta) - (R+L))

I = integrate(a, (theta, 0, np.pi/2))

#%% Problem 3
L = Symbol('L')
x = Symbol('x')
w = Symbol('w')
E = Symbol('E')
I = Symbol('I')
I1 = I/2

M = x*L*w/4
dM = x/2
d = integrate(1/(E*I1)*M*dM, (x, (0, L/4)))
print(d)

M1 = L**2*w/16 + L*w/4*(L/2 - x) - w/2*(L/2 - x)**2
dM1 = (3*L-4*x)/8

d1 = integrate(1/(E*I)*M1*dM1, (x,(L/4,L/2)))
print(simplify(d1))

dt = 2*(d1+d)
print(simplify(dt))

#%% Probelem 4
L = Symbol('L')
x = Symbol('x')
w = Symbol('w')
y = Symbol('y')
E = Symbol('E')
I = Symbol('I')
nu = Symbol('nu')
J = 2*I 
G = E/(2*(1+nu))
Mx = w/2*y**2
dMx = y
d = integrate(1/(E*I)*Mx*dMx, (y,(0,L))) 
print(simplify(d))

T = w*L**2/2
dT = L
My = (w*L)*(L-x)
dMy = L - x
IM = integrate(1/(E*I)*My*dMy,(x,(0,L)))
IT = integrate(1/(G*J)*T*dT,(x,(0,L)))
print(simplify(IM))
print(simplify(IT))
d1 = IM+IT
dt = d + d1
print(simplify(dt))
#%% Problem 5
P = Symbol('P')
Cy = Symbol('Cy')
L = Symbol('L')
x = Symbol('x')



M = 3*L*P/2 - L*Cy - P*x +Cy*x
dM = x - L
d = integrate(M*dM, (x,(0,L/2)))

M1 = L*P - (L - x)*Cy
dM1 = -(L-x)
d1 = integrate(M1*dM1, (x, (L/2, L)))

eqn = Eq(0, d1+d)
print(solve(eqn, Cy))

#%% Problem 6
R = Symbol('R')
P = Symbol('P')
Mb = Symbol('Mb')
theta = Symbol('theta')
Cy = Symbol('Cy')
E = Symbol('E')
I = Symbol('I')
x = 0
M = 2*R*Cy - Mb - Cy*R*(1 - cos(theta))
dM = 2*R - R*(1 - cos(theta))
d = integrate(1/(E*I)*M*dM*R, (theta, (0, pi/2)))
print(simplify(d))

M1 = R*(1 - cos(pi - theta))*Cy
dM1 = R*(1 - cos(pi - theta))
d1 = integrate(1/(E*I)*M1*dM1*R, (theta, (pi/2, pi)))
print(simplify(d1))
eqn1 = Eq(0, d+d1)
ans = solve(eqn1, Cy)
print(ans)

