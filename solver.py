import abc
import torch
import numpy as np
import sde_lib
import random

#@title Solver

class Solver(abc.ABC):
  """The abstract class for a solver algorithm."""

  def __init__(self, sde, score_fn):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE
    self.rsde = sde.reverse(score_fn)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the solver.
    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.
    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

class EulerMaruyamaSolver(Solver):
    def __init__(self, sde, score_fn):
        super().__init__(sde, score_fn)
    
    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        dw = np.sqrt(-dt) * z
        drift, diffusion, _ = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * dw
        return x

class MilsteinSolver(Solver):
    def __init__(self, sde, score_fn):
        super().__init__(sde, score_fn)
    
    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        dw = np.sqrt(-dt) * z
        drift, diffusion, diffusion_grad = self.rsde.sde(x, t)
        x_mean = x + drift * dt 
        c1 = (np.square(dw.cpu()) - dt).to('cuda')
        x = x_mean + diffusion[:, None, None, None] * dw + \
          0.5 * diffusion[:, None, None, None] * diffusion_grad[:, None, None, None] * c1
        return x

class SRK1Solver(Solver):
    def __init__(self, sde, score_fn):
        super().__init__(sde, score_fn)
    
    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        dw = np.sqrt(-dt) * z
        drift, diffusion, _ = self.rsde.sde(x, t)
        x_hat = x + drift * dt + diffusion[:, None, None, None] * np.sqrt(-dt)
        _, diffusion_hat, _ = self.rsde.sde(x_hat, t)
        x_mean = x + drift * dt 
        c1 = (np.square(dw.cpu()) - dt).to('cuda') / np.sqrt(-dt)
        x = x_mean + diffusion[:, None, None, None] * dw + \
            0.5 * (diffusion_hat[:, None, None, None] - diffusion[:, None, None, None]) * c1
        return x
      
class SRK2Solver(Solver):
    def __init__(self, sde, score_fn):
        super().__init__(sde, score_fn)
    
    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        dw = np.sqrt(-dt) * z
        drift, diffusion, _ = self.rsde.sde(x, t)
        x_hat = x + drift * dt + diffusion[:, None, None, None] * dw
        drift_hat, diffusion_hat, _ = self.rsde.sde(x_hat, t)
        x = x + 0.5 * (drift_hat + drift) * dt +  diffusion[:, None, None, None] * dw
        return x

class SIESolver(Solver):
  def __init__(self, sde, score_fn):
        super().__init__(sde, score_fn)
    
  def update_fn(self, x, t):
      dt = -1. / self.rsde.N
      z = torch.randn_like(x)
      dw = np.sqrt(-dt) * z
      drift, diffusion, _ = self.rsde.sde(x, t)
      s = 1 if random.random() < 0.5 else -1
      k1 = drift * dt + diffusion[:, None, None, None] * (dw - s * np.sqrt(-dt))
      drift, diffusion, _ = self.rsde.sde(x + k1, t - dt)
      k2 = drift * dt + diffusion[:, None, None, None] * (dw + s * np.sqrt(-dt))
      x = x + 0.5 * (k1 + k2)
      return x
    
class WeakOrder2RK(Solver):
  def __init__(self, sde, score_fn):
    super().__init__(sde, score_fn)
    
  def update_fn(self, x, t):
      dt = -1. / self.rsde.N
      z = torch.randn_like(x)
      dw = np.sqrt(-dt) * z
      drift, diffusion, _ = self.rsde.sde(x, t)
      x_tilde = x + drift * dt + diffusion[:, None, None, None] * dw
      x_tilde_pos = x + drift * dt + diffusion[:, None, None, None] * np.sqrt(-dt)
      x_tilde_min = x + drift * dt - diffusion[:, None, None, None] * np.sqrt(-dt)
      drift_tilde, diffusion_tilde, _ = self.rsde.sde(x_tilde, t)
      drift_pos, diffusion_pos, _ = self.rsde.sde(x_tilde_pos, t)
      drift_min, diffusion_min, _ = self.rsde.sde(x_tilde_min, t)
      x = x + 0.5 * (drift + drift_tilde) * dt \
          + 0.25 * (diffusion_pos[:, None, None, None] + diffusion_min[:, None, None, None] + 2 * diffusion[:, None, None, None]) * dw \
          + 0.25 / np.sqrt(-dt) * (diffusion_pos[:, None, None, None] - diffusion_min[:, None, None, None]) * (np.square(dw.cpu()) + dt).to('cuda')
      return x
