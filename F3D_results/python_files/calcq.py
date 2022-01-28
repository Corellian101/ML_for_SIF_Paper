import math

def EllipseCircumference(a, b):
   """
   Compute the circumference of an ellipse with semi-axes a and b.
   Require a >= 0 and b >= 0.  Relative accuracy is about 0.5^53.
   """
   import math
   x, y = max(a, b), min(a, b)
   digits = 53; tol = math.sqrt(math.pow(0.5, digits))
   if digits * y < tol * x: return 4 * x
   s = 0; m = 1
   while x - y > tol * y:
      x, y = 0.5 * (x + y), math.sqrt(x * y)
      m *= 2; s += m * math.pow(x - y, 2)
   return math.pi * (math.pow(a + b, 2) - s) / (x + y)

def shapefactor(a,c):
    P = EllipseCircumference(a,c)
    return (P/(4*c))**2

def empirical_shape_factor(a, c):
    if a < c:
	Q = 1 + 1.464*(a/c)**(1.65)
	return Q
    elif a > c:
	Q = 1 + 1.464*(c/a)**(1.65)
	return Q
    elif a == c:
	Q = math.pi**2/4
	return Q
