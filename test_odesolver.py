from odesolver import ForwardEuler, RRK, DiffEqSolver
import matplotlib.pyplot as plt
import numpy as np
import torch


# Harmonic Oscillator with Quartic Entropy/Energy
class HarmonicOscillator:
    def f(w, t=0.0):
        return torch.tensor([-w[1], w[0]])

    def eta(w):
        return torch.tensor(
            [
                w[0] * w[0] * w[0] * w[0]
                + 2 * w[0] * w[0] * w[1] * w[1]
                + w[1] * w[1] * w[1] * w[1]
            ]
        )

    def deta(w):
        return torch.tensor(
            [
                4 * w[0] * w[0] * w[0] + 4 * w[0] * w[1] * w[1],
                4 * w[1] * w[1] * w[1] + 4 * w[1] * w[0] * w[0],
            ]
        )

    def u_analytical(t):
        w0 = torch.cos(t)
        w1 = torch.sin(t)
        return torch.tensor([w0, w1])


## Solve forward

print(" \n-- Solving forward --\n ")


TEND = 1.0

u0 = torch.tensor([1.0, 0.0])
dt = 0.1
t0 = torch.tensor([0.0])
tf = torch.tensor([TEND])

fe = "FE"
trap = "SSP22"
rk4 = "RK44"
# ode_solver = RRK(h_max=dt, rkm=rk4, relaxation=True, rescale_step=True, store_sol=True)
ode_solver = DiffEqSolver(h_max=dt, store_sol=True)

tt, uu = ode_solver.solve(u0, t0, tf, HarmonicOscillator.f)

# print('t history = ', tt)

# H = torch.cat([HarmonicOscillator.eta(uu[:,i]) for i in range(uu.shape[1])])
# plt.plot(tt, H - H[0]);
# plt.xlabel("$t$"); plt.ylabel("$\eta(u_\mathrm{num}(t)) - \eta(u_0)$"); plt.xlim(tt[0], tt[-1]);

print("u(t_0) = ", uu[:, 0])
print("u(t_f) = ", uu[:, -1])

print(
    "Error at tf: %.3e"
    % np.linalg.norm(uu[:, -1] - HarmonicOscillator.u_analytical(tt[-1]))
)
print(
    "Pointwise mean error: %.3e"
    % np.mean(
        [
            np.linalg.norm(uu[:, i] - HarmonicOscillator.u_analytical(tt[i]))
            for i in range(tt.size(dim=0))
        ]
    )
)

print(" \n-- Solving backward --\n ")

## Solve backward

u0 = uu[:, -1]
tf = torch.tensor([0.0])
t0 = torch.tensor([TEND])

tt, uu_bwd = ode_solver.solve(u0, t0, tf, HarmonicOscillator.f)

# print('t history = ', tt)

# H = torch.cat([HarmonicOscillator.eta(uu[:,i]) for i in range(uu.shape[1])])
# plt.plot(tt, H - H[0]);
# plt.xlabel("$t$"); plt.ylabel("$\eta(u_\mathrm{num}(t)) - \eta(u_0)$"); plt.xlim(tt[0], tt[-1]);

print("u(t_0) = ", uu_bwd[:, -1])
print("u(t_f) = ", uu_bwd[:, 0])
print(
    "Error at t0: %.3e"
    % np.linalg.norm(uu_bwd[:, -1] - HarmonicOscillator.u_analytical(tt[-1]))
)
print(
    "Pointwise mean error: %.3e"
    % np.mean(
        [
            np.linalg.norm(uu_bwd[:, i] - HarmonicOscillator.u_analytical(tt[i]))
            for i in range(tt.size(dim=0))
        ]
    )
)
