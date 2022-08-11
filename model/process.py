import numpy as np
import scipy

# CGMY process
def cgmy_lambda_p_fun(sig, nu, theta):
  temp = theta / (sig**2)
  return np.sqrt(temp**2+2/(sig**2*nu)) - temp
def cgmy_lambda_n_fun(sig, nu, theta):
  temp = theta / (sig**2)
  return np.sqrt(temp**2+2/(sig**2*nu)) + temp
def cgmy_k_fun(lambda_, nu, Y, x):
  abs_ = np.abs(x)
  return np.exp(-lambda_*abs_)/nu/abs_**(1+Y)
def cgmy_sig2_eps_fun(sig, nu, theta, Y, Dx_p, Dx_n = None):
  if Dx_n is None: Dx_n = Dx_p
  paraC = 1/nu
  paraG = cgmy_lambda_n_fun(sig, nu, theta)
  paraM = cgmy_lambda_p_fun(sig, nu, theta)
  paraY = Y
  return paraC * pow(paraG,paraY-2) * (igamma(2-paraY,0)-igamma(2-paraY,paraG*Dx_n)) + paraC * pow(paraM,paraY-2) * (igamma(2-paraY,0)-igamma(2-paraY,paraM*Dx_p))
def cgmy_omg_eps_fun(sig, nu, theta, Y, Dx_p, Dx_n = None):
  if Dx_n is None: Dx_n = Dx_p
  paraC = 1/nu
  paraG = cgmy_lambda_n_fun(sig, nu, theta)
  paraM = cgmy_lambda_p_fun(sig, nu, theta)
  paraY = Y
  return paraC * (pow(paraM,paraY)*igamma(-paraY,paraM*Dx_p)-pow(paraM-1,paraY)*igamma(-paraY,(paraM-1)*Dx_p)+
            pow(paraG,paraY)*igamma(-paraY,paraG*Dx_n)-pow(paraG+1,paraY)*igamma(-paraY,(paraG+1)*Dx_n))


# NIG process
def nig_k_fun(alpha, beta, delta, x):
  abs_ = np.abs(x)
  return delta*alpha/np.pi*np.exp(beta*abs_)/abs_*scipy.special.k1(alpha*abs_)
def nig_sig2_eps_fun(alpha, beta, delta, Dx_p, Dx_n=None):
  if Dx_n is None:
    Dx_n = Dx_p 
  int_p = scipy.integrate.quadrature(lambda x: x**2*nig_k_fun(alpha, beta, delta, x), a=0, b=Dx_p,tol = 0, rtol = 1e-6,maxiter=500)[0]
  int_n = scipy.integrate.quadrature(lambda x: x**2*nig_k_fun(alpha, -beta, delta, x), a=0, b=Dx_n,tol = 0, rtol = 1e-6,maxiter=500)[0]
  return int_p + int_n
def nig_omg_eps_fun(alpha, beta, delta, Dx_p, Dx_n=None):
  if Dx_n is None:
    Dx_n = Dx_p 
  int_p = scipy.integrate.quadrature(lambda x: (1-np.exp(x))*nig_k_fun(alpha, beta, delta, x), a=Dx_p, b=20,tol = 0, rtol = 1e-6,maxiter=2000)[0]
  int_n = scipy.integrate.quadrature(lambda x: (1-np.exp(-x))*nig_k_fun(alpha, -beta, delta, x), a=Dx_n, b=20,tol = 0, rtol = 1e-6,maxiter=2000)[0]
  return int_p + int_n

# Merton's jump diffusion model
def merton_k_fun(lam, alpha, delta, x):
  return lam/np.sqrt(2*np.pi)/delta * np.exp(-(x-alpha)**2/2/delta**2)
def merton_sig2_eps_fun(sig):
  return sig**2
def merton_omg_eps_fun(lam, alpha, delta): 
  return -lam*(np.exp(alpha+delta**2/2)-1)

#  The Kou’s double exponential jump diffusion model
def kou_k_fun(lam, p, eta, x):
  return lam*p*eta*np.exp(-eta*abs(x))
def kou_sig2_eps_fun(sig):
  return sig**2
def kou_omg_eps_fun(lam, p, eta1, eta2): 
  return -lam*(p*eta1/(eta1-1)+(1-p)*eta2/(eta2+1)-1)

## incomplete gamma function for CGMY 
## reference: a computational procedure for incomplete gamma functions, W. Gautschi, 1979 
def igamma(a, x):
    if x<0:
      raise Exception('x<0!')
    if x==0: 
      return scipy.special.gamma(a)
    x0 = 1.5
    EPS = 1e-6
    EULER =.577215664901532860606512
    if x>=1/4:
      alpha=x+1/4
    else:
      alpha=np.log(1/2)/np.log(x)
    if a>alpha:
      t=1
      s=t
      fac=x**a*np.exp(-x)/a
      eps1=np.abs(EPS/fac)
      k=0
      while np.abs(t)>eps1:
          k=k+1
          t=t*x/(a+k)
          s=s+t
      s=scipy.special.gamma(a)-s*fac
    elif x>x0:
      t=1
      r=1
      s=t
      fac=x**a*np.exp(-x)/(x-a+1)
      eps1=np.abs(EPS/fac)
      k=0
      while np.abs(t)>eps1:
          k=k+1
          ak=k*(a-k)/(x+2*k-1-a)/(x+2*k+1-a)
          r=1/(1+ak*r)
          t=t*(r-1)
          s=s+t	
      s=s*fac
    elif (a<-1/2):
      m=int(np.floor(1/2-a))
      a=a+m
      if a==0:
          u=-EULER-np.log(x)
      else:
          u=(scipy.special.gamma(1+a)-x**a)/a
      p=a*x
      q=a+1
      r=a+3
      t=1
      v=t
      fac=x**(a+1)/(a+1)
      eps1=np.abs(EPS/fac)
      while np.abs(t)>eps1:
        p=p+x
        q=q+r
        r=r+2
        t=-p*t/q
        v=v+t		   
      v=v*fac
      s=u+v
      s=np.exp(x)*pow(x,-a)*s
      for k in range(1,m+1):
          s=(1-x*s)/(k-a)
      s=pow(x,a-m)*np.exp(-x)*s
    else:
      if a==0:
          u=-EULER-np.log(x)
      else:
          u=(scipy.special.gamma(1+a)-pow(x,a))/a
      p=a*x
      q=a+1
      r=a+3
      t=1
      v=t
      fac=pow(x,a+1)/(a+1)
      eps1=np.abs(EPS/fac)
      while np.abs(t)>eps1:
        p=p+x
        q=q+r
        r=r+2
        t=-p*t/q
        v=v+t   
      v=v*fac
      s=u+v
    return s
