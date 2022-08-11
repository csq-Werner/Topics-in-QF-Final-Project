import numpy as np
from model.benchmark import option_CF
## European option reference: option valuation using the fast Fourier transform, P. Carr and D.B. Madan, 1999 
def option_eu(model,cp,S,K,T,r,q,**paras):
    s0 = K
    if cp not in ['call','put']: raise ValueError
    if cp=='call':
      damp = 0.3
    elif cp=='put':
      damp = -1.3
    eta = 0.05

    N = 2**16
    lda_eta = 2*np.pi/N
    lda = lda_eta/eta
    bB = np.log(K)-N*lda/2

    jJ = np.arange(N)+1
    vj = (jJ-1)*eta
    m = np.arange(N)+1
    km = bB + (m-1)*lda
    psi_vj = option_CF(model,vj -(damp+1)*1j, s0, r, q, T, **paras) / ((damp + 1j*vj)*(damp + 1 + 1j*vj))

    diracdelt = (jJ==1)
    wj = (eta/3)*(3 + (-1)**jJ - diracdelt)
    xx = np.exp(-1j*bB*vj)*psi_vj*wj
    zz = np.fft.fft(xx)

    multiplier = np.exp(-damp*km)/np.pi
    zz2 = multiplier*zz
    out=S/s0*(np.exp(-r*T)*np.interp(K*s0/S, np.exp(km), np.real(zz2)))
    return out
## American option reference: a fast and accurate FFT-based method for pricing early-exercise options under Lévy processes, R. Lord et al., 2008
def option_am_paras(model,cp,S,K,T,r,q,damp,M,N,eta, **paras):
    s0 = K
    lda_eta = 2*np.pi/N
    lda = lda_eta/eta
    bB = np.log(K)-N*lda/2

    jJ = np.arange(N)+1
    vj = (jJ-1)*eta - N/2 * eta
    m = np.arange(N)+1
    xm = bB + (m-1)*lda
    wj = np.ones(N)
    wj[0] = 1/2
    wj[-1] = 1/2
    multi = (-1)**(jJ-1)
    mul1 = np.exp(damp*xm) * wj * multi
    psi_vj = option_CF(model,-vj+(damp)*1j, 1, r, q, T/M, **paras)
    mul2 = np.exp(-r*T/M - damp*(xm)) * multi

    if cp=='call':
        z0 = np.maximum(np.exp(xm)-K,0)
    elif cp=='put':
        z0 = np.maximum(K-np.exp(xm),0)
    else: raise ValueError
    
    z = z0
    for l in range(M):
        z1 = z * mul1
        fftv = np.fft.ifft(z1)        
        z = np.fft.fft(psi_vj * fftv) 
        z = z * mul2        
        z=np.maximum(z,z0)

    return np.interp(S, np.exp(xm), np.real(z))

def option_am(model,cp, S, K,T,r,q,**paras):
    if cp=='call':
        damp = -1.2
    elif cp=='put':
        damp = 1.2
    else: raise ValueError 
    eta = 0.03
    dd=2
    l=16
    out_1=option_am_paras(model,cp,S,K,T,r,q,damp,2**(dd+3),2**l,eta, **paras)
    out_2=option_am_paras(model,cp,S,K,T,r,q,damp,2**(dd+2),2**l,eta, **paras)
    out_3=option_am_paras(model,cp,S,K,T,r,q,damp,2**(dd+1),2**l,eta, **paras)
    out_4=option_am_paras(model,cp,S,K,T,r,q,damp,2**dd,2**l,eta, **paras)
    return ((64*out_1-56*out_2+14*out_3-out_4)/21)
