"""Regularized probability offsets; the caller must supply only matured labels."""
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize,minimize_scalar

def fit_binary_offset(probabilities,labels):
    p=np.asarray(probabilities,dtype=float);y=np.asarray(labels,dtype=float)
    if p.shape!=y.shape or p.ndim!=1 or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():raise ValueError('Expected matched finite probabilities and binary labels')
    p=np.clip(p,1e-8,1-1e-8);z=np.log(p/(1-p))
    def loss(b):return float(np.sum(np.logaddexp(0,z+b)-y*(z+b))+.5*b*b)
    result=minimize_scalar(loss,bounds=(-4,4),method='bounded',options={'xatol':1e-9,'maxiter':200})
    if not result.success:raise RuntimeError('Binary calibration did not converge')
    return float(result.x)

def binary_offset(probability,bias):
    p=np.clip(np.asarray(probability,dtype=float),1e-8,1-1e-8);return expit(np.log(p/(1-p))+bias)

def fit_vector_offset(probabilities,labels):
    p=np.asarray(probabilities,dtype=float);y=np.asarray(labels,dtype=int)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)!=len(y) or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.allclose(p.sum(axis=1),1) or not np.isin(y,[0,1,2]).all():raise ValueError('Expected three-class probability rows and labels')
    z=np.log(np.clip(p,1e-8,1));actual=np.eye(3)[y]
    def fun(b):
        q=z+np.r_[0.,b];norm=logsumexp(q,axis=1);prob=np.exp(q-norm[:,None]);loss=float(-np.sum(q[np.arange(len(y)),y]-norm)+.5*np.sum(b*b));gradient=(prob-actual).sum(axis=0)[1:]+b;return loss,gradient
    result=minimize(fun,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-4,4)]*2,options={'ftol':1e-12,'gtol':1e-8,'maxiter':200})
    # For this strictly convex objective, a projected first-order residual also
    # certifies a bounded optimum when a line search stops at floating precision.
    _,gradient=fun(result.x);projected=result.x-np.clip(result.x-gradient,-4,4)
    if not result.success and np.max(np.abs(projected))>1e-5:raise RuntimeError('Vector calibration did not converge: '+str(result.message)+'; projected_gradient='+str(float(np.max(np.abs(projected)))))
    return result.x

def vector_offset(probability,bias):
    p=np.asarray(probability,dtype=float);z=np.log(np.clip(p,1e-8,1))+np.r_[0.,np.asarray(bias,dtype=float)];return np.exp(z-logsumexp(z,axis=-1,keepdims=True))
