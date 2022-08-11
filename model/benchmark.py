import numpy as np
import scipy
"""## Benchmark"""

def cgmy_CF(u, s0, r, q, t, sigma, nu, theta, Y):
    if Y==0:
        if nu == 0:
            mu = np.log(s0) + (r-q - theta -0.5*sigma**2)*t
            phi  = np.exp(1j*u*mu) * np.exp((1j*theta*u-0.5*sigma**2*u**2)*t)
        else:
            mu  = np.log(s0) + (r-q + np.log(1-theta*nu-0.5*sigma**2*nu)/nu)*t
            phi = np.exp(1j*u*mu) * ((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-t/nu))
    else:
        G = (theta**2/sigma**4+2/sigma**2/nu)**(1/2)+theta/sigma**2
        M = (theta**2/sigma**4+2/sigma**2/nu)**(1/2)-theta/sigma**2
        C = 1/nu
        coef = C*scipy.special.gamma(-Y)
        mu  = np.log(s0) + (r-q - coef*((G+1)**Y-G**Y+(M-1)**Y-M**Y))*t
        phi = np.exp(1j*u*mu) * np.exp(coef*((G+1j*u)**Y-G**Y+(M-1j*u)**Y-M**Y)*t)
    return phi

def nig_CF(u, s0, r, q, t, alpha, beta, delta):
    mu = np.log(s0) + (r-q + delta*(np.sqrt(alpha**2-(beta+1)**2)-np.sqrt(alpha**2-beta**2)))*t
    phi  = np.exp(1j*u*mu) * np.exp(-delta*(np.sqrt(alpha**2-(beta+u*1j)**2)-np.sqrt(alpha**2-beta**2))*t)
    return phi

def merton_CF(u, s0, r, q, t, sig, lam, alpha, delta):
    mu = np.log(s0) + (r-q - sig**2/2 - lam * (np.exp(alpha+delta**2/2)-1))*t
    phi  = np.exp(1j*u*mu) * np.exp(- t*u**2*sig**2/2 + lam * (np.exp(u*alpha*1j-u**2*delta**2/2)-1)*t)
    return phi

def kou_CF(u, s0, r, q, t, sig, lam, p, eta1, eta2):
    mu = np.log(s0) + (r-q - sig**2/2 - lam * (p*eta1/(eta1-1)+(1-p)*eta2/(eta2+1)-1))*t
    phi  = np.exp(1j*u*mu) * np.exp(- t*u**2*sig**2/2 + lam * (p*eta1/(eta1-u*1j)+(1-p)*eta2/(eta2+u*1j)-1)*t)
    return phi

def option_CF(model,u, s0, r, q, T, **paras):
  if model == 'VG': 
    return cgmy_CF(u, s0, r, q, T, paras['sig'], paras['nu'], paras['theta'],0)
  elif model == 'CGMY': 
    return cgmy_CF(u, s0, r, q, T, paras['sig'], paras['nu'], paras['theta'], paras['Y'])
  elif model == 'NIG': 
    return nig_CF(u, s0, r, q, T, paras['alpha'], paras['beta'], paras['delta'])
  elif model == 'Merton':
    return merton_CF(u, s0, r, q, T, paras['sig'], paras['lam'], paras['alpha'], paras['delta'])
  elif model == 'Kou':
    return kou_CF(u, s0, r, q, T, paras['sig'], paras['lam'], paras['p'], paras['eta1'], paras['eta2'])
  else: raise NotImplementedError
