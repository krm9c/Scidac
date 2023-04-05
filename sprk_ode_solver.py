import math
import torch
import numpy as np

class SprkAdjoint:
  def __init__(self) -> None:
    self.stages = 0
    self.order = 0
    
  def solve_forward(self, z0, t0, t1, f):
    pass
  
  def solve_backward(self, z0, t0, t1, f):
    pass
    
class Leapfrog(SprkAdjoint):
  def __init__(self) -> None:
    self.stages = 2
    self.order = 2
    self.fcoeffs = torch.tensor([0.5, 0.5], dtype=torch.float64)
    self.bcoeffs = torch.tensor([0.0, 1.0], dtype=torch.float64)
  
  def solve_forward(self, z0, t0, t1, f):
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    assert t1 > t0, 'solve_forward only solves forward in time'

    h = abs(t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
      k1 = f(z, t)
      X1 = z + self.fcoeffs[0]*k1*h
      k2 = f(z + self.fcoeffs[0]*h*k1, t + self.fcoeffs[0]*h)
      X2 = z + h * (self.fcoeffs[0]*k1 + self.fcoeffs[1]*k2)
      z = X2
      t = t + h

    return z, X1, X2

  def solve_backward(self, z0, t0, t1, f):
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    assert t0 > t1, 'solve_backward only solves backward in time'

    h = abs(t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
      # We first solve forward over the interval that we want to integrate backward in order
      # to get the intermediate stage values that are needed in the backwards step.
      _, X1, X2 = self.solve_forward(z0, t-h, t, f)
      # Now do the backward step using the intermediate stage values
      z = z - h * (self.bcoeffs[0] * X1 + self.bcoeffs[1] * X2)
      t = t - h

    return z
       

class McLachlan4(SprkAdjoint):
    def __init__(self) -> None:
      self.stages = 4
      self.order = 4
      a0 = torch.float64(0.515352837431122936)
      a1 = -torch.float64(0.085782019412973646)
      a2 = torch.float64(0.441583023616466524)
      a3 = torch.float64(0.128846158365384185)
      b0 = torch.float64(0.134496199277431089)
      b1 = -torch.float64(0.224819803079420806)
      b2 = torch.float64(0.756320000515668291)
      b3 = torch.float64(0.33400360328632142)
      self.fcoeffs = torch.tensor([a0, a1, a2, a3], dtype=torch.float64)
      self.bcoeffs = torch.tensor([b0, b1, b2, b3], dtype=torch.float64)
