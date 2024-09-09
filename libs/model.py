# ------------------------------------------------------------------------------------------------------------------------------------------------
# imports
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from functools import partial
import jax.nn as jnn
import numpy as np_
from jax import lax
import diffrax
from libs.utils import sp_matmul
import equinox as eqx

## Train now a CNN and test the trainer and then, the older model

from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping



# ---------------------------------------------------------------------------------------------------------
## Simple feedforward NN
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    initializer: None 
    def __init__(self, in_size, out_size, key):
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (out_size, in_size))
        self.bias = self.initializer(bkey, (1, out_size))
    def __call__(self, x):
        # print(self.weight.shape, x.shape)
        x = x @ self.weight.T
        # print(x.shape)
        x = x+ self.bias
        # print(x.shape)
        return x
    
    
# #---------------------------------------------------------------------------------------------------------
## MLP layer
# #---------------------------------------------------------------------------------------------------------
class MLP(eqx.Module):
    input_layer: None
    feed_layers: list
    output_layers: None
    def __init__(self, key, input_dim=2, out_dim=2, n_layers=2, hln=200):
        self.input_layer = eqx.nn.Linear(
            input_dim, hln, key=jax.random.PRNGKey(key))
        self.feed_layers = [eqx.nn.Linear(
            hln, hln, key=jax.random.PRNGKey(i)) for i in range(1, n_layers-2)]
        self.output_layers =  eqx.nn.Linear( hln, out_dim, key=jax.random.PRNGKey(n_layers-1))

    def __call__(self, x, actfunc__=jax.nn.tanh, outfunc=None):
        x = actfunc__(self.input_layer(x))
        for element in self.feed_layers:
            x = actfunc__(element(x))
        if outfunc is None:
            return self.output_layers(x)
        else:
            return outfunc(self.output_layers(x))
    
     
import numpy as np
class Func(eqx.Module):
    # mlp: MLP
    scale: MLP
    coeff: MLP
    # mlp3: eqx.nn.MLP
    # mlp4: eqx.nn.MLP
    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.scale = MLP(key=key, input_dim=(data_size), out_dim=100, n_layers=depth, hln=width_size)
        self.coeff = MLP(key=key, input_dim=(data_size), out_dim=100, n_layers=depth, hln=width_size)
        
    def __call__(self, t, y, args):
        # print(y.shape)
        exponents= (-1*self.scale(y, outfunc=jnp.exp))+self.coeff(y*(t) ) 
        # print(exponents.shape)
        out = a1*jnp.exp(-a2*alpha(hbaromega, t) ) +jnp.mean(jnp.exp( exponents ) )
        # print("before concat", out.shape, y[1].shape)
        out = jnp.array([out, y[1]])
        # print("after concat", out.shape)
        return out
    
        # return  self.mlp(y)*jnp.exp(-1*self.mlp2(y, outfunc=jnp.exp)*t  ) \
        #     +   self.mlp2(y)*jnp.exp(-1*self.mlp2(y, outfunc=jnp.exp)*t**(9/10) ) \
        #     +   self.mlp(y)*jnp.exp(-1*self.mlp2(y, outfunc=jnp.exp)*t**(8/10) ) \
        #     +   self.mlp2(y)*jnp.exp(-1*self.mlp2(y, outfunc=jnp.exp)*t**(7/10) )\
        #     +   self.mlp(y)*jnp.exp(-1*self.mlp3(y, outfunc=jnp.exp)*t**(6/10) ) \
        #     +   self.mlp2(y)*jnp.exp(-1*self.mlp4(y, outfunc=jnp.exp)*t**(5/10) ) \
        #     +   self.mlp(y)*jnp.exp(-1*self.mlp3(y, outfunc=jnp.exp)*t**(4/10) )\
        #     +   self.mlp2(y)*jnp.exp(-1*self.mlp4(y, outfunc=jnp.exp)*t**(3/10) )\
        #     +   self.mlp(y)*jnp.exp(-1*self.mlp3(y, outfunc=jnp.exp)*t**(2/10) ) \
        #     +   self.mlp2(y)*jnp.exp(-1*self.mlp4(y, outfunc=jnp.exp)*t**(1/10) ) 
        #         # +0.25*self.mlp(y)*(jnp.exp(-5*t))\
        #         # +0.25*self.mlp2(y)*(jnp.exp(-2*t))\
        #         # +0.25*self.mlp3(y)*(jnp.exp(-t))



class NeuralODE(eqx.Module):
    func: Func
    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        # print("within the neural ode", ts.shape, y0.shape)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
            saveat=diffrax.SaveAt(ts=ts),
        )
        ys = solution.ys
        return ys