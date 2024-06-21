from abc import ABC, abstractmethod
import math
import torch
from nodepy import rk
from torchdiffeq import odeint


class ODESolver(ABC):
    @abstractmethod
    def solve(self, z0, t0, tf, f):
        pass


class ForwardEuler(ODESolver):
    def __init__(self, h_max, store_sol=False):
        self.h_max = h_max
        self.store_sol = store_sol

    def solve(self, z0, t0, tf, f):
        device = z0.device
        h_max = self.h_max
        n_steps = math.ceil((abs(tf - t0) / h_max).max().item())
        dt = (tf - t0) / n_steps
        t = t0.detach().clone()
        z = z0.detach().clone()
        if self.store_sol:
            tt = [t.detach().clone()]
            # zz = torch.zeros([len(z0), n_steps + 1]).to(device)
            zz = torch.zeros([*z0.shape, int(abs(tf - t) / abs(dt) * 2.5) + 10000]).to(
                device
            )
            zz[:, 0] = z.detach().clone()
        for ii in range(n_steps):
            z = z + dt * f(z, t)
            t = t + dt
            if self.store_sol:
                zz[:, ii] = z.detach().clone()
                tt.append(t.detach().clone())
        if self.store_sol:
            return torch.cat(tt).to(device), zz[:, : ii + 1]
        else:
            return z


class RRK(ODESolver):
    def __init__(
        self,
        h_max,
        rkm="RK44",
        relaxation=True,
        rescale_step=True,
        store_sol=False,
    ):
        self.rkm = rk.loadRKM(rkm).__num__()
        self.h_max = h_max
        self.relaxation = relaxation
        self.rescale_step = rescale_step
        self.store_sol = store_sol

    def solve(self, z0, t0, tf, f):
        n_steps = math.ceil((abs(tf - t0) / self.h_max).max().item())
        dt = (tf - t0) / n_steps
        return RRK.advance(
            self.rkm,
            dt,
            f,
            z0=z0,
            t0=t0,
            tf=tf,
            relaxation=self.relaxation,
            rescale_step=self.rescale_step,
            store_sol=self.store_sol,
        )

    def advance(
        rkm,
        dt,
        f,
        z0=[1.0, 0],
        t0=0.0,
        tf=1.0,
        relaxation=True,
        rescale_step=True,
        store_sol=False,
        debug=False,
        gammatol=0.1,
        print_gamma=False,
        one_step=False,
        dissip_factor=1.0,
    ):
        """
        Original source attibution:
            Copyright (c) 2019 David I. Ketcheson and Hendrik Ranocha.

        Modified by Cody Balos.

        Relaxation Runge-Kutta method implementation.

        Options:

            rkm: Base Runge-Kutta method, in Nodepy format
            dt: time step size
            f: RHS of ODE system
            z0: Initial data
            t0: starting solution time
            tf: final solution time
            relaxation: if True, use relaxation method.  Otherwise, use vanilla RK method.
            rescale_step: if True, new time step is t_n + \gamma dt
            debug: output some additional diagnostics
            gammatol: Fail if abs(1-gamma) exceeds this value

        """
        device = z0.device
        t = t0.detach().clone()
        z = z0.detach().clone()
        if store_sol:
            # We pre-allocate extra space because if rescale_step==True then
            # we don't know exactly how many steps we will take.
            zz = torch.zeros([*z0.shape, int(abs(tf - t) / abs(dt) * 2.5) + 10000]).to(
                device
            )
            zz[:, 0] = z.detach().clone()
            tt = [t.detach().clone()]
        ii = 0
        s = len(rkm)
        A = torch.tensor(rkm.A).to(device)
        b = torch.tensor(rkm.b).to(device)
        y = torch.zeros((s, *z0.shape)).to(device)
        max_gammam1 = 0.0
        gams = []

        while t < tf if tf > t0 else t > tf:

            if (t + dt - tf) * dt > 0.0:
                dt = tf - t  # Hit final time exactly

            for i in range(s):
                y[i, :] = z.detach().clone()
                for j in range(i):
                    y[i, :] += A[i, j] * dt * f(y[j, :], t)

            F = [f(y[i, :], t) for i in range(s)]

            if relaxation:
                numer = 2 * sum(
                    b[i] * A[i, j] * torch.tensordot(F[i], F[j], dims=F[i].dim())
                    for i in range(s)
                    for j in range(s)
                )
                denom = sum(
                    b[i] * b[j] * torch.tensordot(F[i], F[j], dims=F[i].dim())
                    for i in range(s)
                    for j in range(s)
                )
                if denom != 0:
                    gam = numer / denom
                else:
                    gam = 1.0
            else:  # Use standard RK method
                gam = 1.0

            if print_gamma:
                print(gam)

            # TODO(CJB): when it is standard RK, shouldn't this be disabled?
            if torch.abs(gam - torch.tensor([1.0]).to(device)) > gammatol:
                print(gam)
                raise Exception("The time step is probably too large.")

            z = z + dissip_factor * gam * dt * sum([b[j] * F[j] for j in range(s)])
            if (t + dt - tf) * dt and rescale_step:
                t += dissip_factor * gam * dt
            else:
                t += dt
            ii += 1
            if store_sol:
                tt.append(t.detach().clone())
                zz[:, ii] = z.detach().clone()
            if debug:
                gm1 = torch.abs(1.0 - gam)
                max_gammam1 = max(max_gammam1, gm1)
                gams.append(gam)

            if one_step:
                return z, gam

        if debug:
            print(max_gammam1)
            return torch.cat(tt), zz[:, : ii + 1], torch.cat(gams)
        elif store_sol:
            return torch.cat(tt), zz[:, : ii + 1]
        else:
            return z


class DiffEqSolver(ODESolver):
    def __init__(self, h_max, store_sol=False, rtol=1e-6, atol=1e-8):
        self.h_max = h_max
        self.store_sol = store_sol
        self.rtol = rtol
        self.atol = atol

    def solve(self, z0, t0, tf, f):
        if self.store_sol:
            if tf > t0:
                t = torch.arange(t0[0], tf[0] + self.h_max / 2, self.h_max)
            else:
                t = torch.arange(t0[0], tf[0] + self.h_max / 2, -self.h_max)
        else:
            t = torch.tensor([tf])
        sol = odeint(lambda t, y: f(y, t), z0, t, rtol=self.rtol, atol=self.atol)
        if self.store_sol:
            return t, sol.T
        else:
            return sol.squeeze()
