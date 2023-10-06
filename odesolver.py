"""
MIT License

Copyright (c) 2019 David I. Ketcheson and Hendrik Ranocha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod
import math
import torch
import numpy as np
from nodepy import rk
from scipy.optimize import root, fsolve, newton, brentq, bisect

class ODESolver(ABC):
    
    @abstractmethod
    def solve(self, z0, t0, tf, f):
        pass
    

class ForwardEuler(ODESolver):
    def __init__(self, h_max):
        self.h_max = h_max

    def solve(self, z0, t0, tf, f):
        h_max = self.h_max
        n_steps = math.ceil((abs(tf - t0)/h_max).max().item())
        h = (tf - t0)/n_steps
        t = t0
        tt = [t]
        z = z0
        for i_step in range(n_steps):
            z = z + h * f(z, t)
            t = t + h
            tt.append(t)
        return tt, z


class RRK(ODESolver):
    def __init__(self, h_max, rkm=rk.loadRKM('RK44').__num__(), relaxation=True, rescale_step=True):
        self.rkm = rkm
        self.h_max = h_max

    def solve(self, z0, t0, tf, f):
        n_steps = math.ceil((abs(tf - t0)/self.h_max).max().item())
        dt = (tf - t0)/n_steps
        return RRK.advance(self.rkm, dt, f, w0=z0, t_init=t0, t_final=tf)
    
    def advance(rkm, dt, f, w0=[1.,0], t_init=0., t_final=1., relaxation=True, 
                rescale_step=True, debug=False, gammatol=0.1, print_gamma=False,
                one_step=False, dissip_factor=1.0):
        """
        Relaxation Runge-Kutta method implementation.
        
        Options:
        
            rkm: Base Runge-Kutta method, in Nodepy format
            dt: time step size
            f: RHS of ODE system
            w0: Initial data
            t_init: starting solution time
            t_final: final solution time
            relaxation: if True, use relaxation method.  Otherwise, use vanilla RK method.
            rescale_step: if True, new time step is t_n + \gamma dt
            debug: output some additional diagnostics
            gammatol: Fail if abs(1-gamma) exceeds this value
            
        """
        w = w0.detach().clone()
        t = t_init
        # We pre-allocate extra space because if rescale_step==True then
        # we don't know exactly how many steps we will take.
        ww = torch.zeros([len(w0),int(abs(t_final-t)/abs(dt)*2.5)+10000])
        ww[:,0] = w.detach().clone()
        tt = [t.detach().clone()]
        ii = 0
        s = len(rkm)
        b = rkm.b
        y = torch.zeros((s,len(w0)))
        max_gammam1 = 0.
        gams = []
        
        while t < t_final if t_final > t_init else t > t_final:
            
            if (t+dt - t_final)*dt > 0.0:
                dt = t_final - t # Hit final time exactly
            
            for i in range(s):
                y[i,:] = w.detach().clone()
                for j in range(i):
                    y[i,:] += rkm.A[i,j]*dt*f(y[j,:])
                    
            F = [f(y[i,:]) for i in range(s)]
            
            if relaxation:
                numer = 2*sum(b[i]*rkm.A[i,j]*torch.dot(F[i],F[j]) \
                                    for i in range(s) for j in range(s))
                denom = sum(b[i]*b[j]*torch.dot(F[i],F[j]) for i in range(s) for j in range(s))
                if denom != 0:
                    gam = numer/denom
                else:
                    gam = 1.
            else:  # Use standard RK method
                gam = 1.
            
            if print_gamma:
                print(gam)
            
            if torch.abs(gam-1.) > gammatol:
                print(gam)
                raise Exception("The time step is probably too large.")
            
            w = w + dissip_factor*gam*dt*sum([b[j]*F[j] for j in range(s)])
            if (t+dt - t_final)*dt and rescale_step:
                t += dissip_factor*gam*dt
            else:
                t += dt
            ii += 1
            tt.append(t.detach().clone())
            ww[:,ii] = w.detach().clone()
            if debug:
                gm1 = torch.abs(1.-gam)
                max_gammam1 = max(max_gammam1,gm1)
                gams.append(gam)
                
            if one_step:
                return w, gam
        
        if debug:
            print(max_gammam1)
            return torch.cat(tt), ww[:, :ii+1], torch.cat(gams)
        else:
            return torch.cat(tt), ww[:, :ii+1]
    
