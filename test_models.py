



#-----------------------------------------------------
# Imports
import argparse
import os
import signal
import sys
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import jax
import jax.tree_util as tree
import numpy as np
import jax.numpy as jnp
from jaxopt import OptaxSolver
import jax.random as jr
import optax

#-------------------------------------------------------
## Regular Linear layer in equinox
from lib.model import *
from lib.utils import *
from lib.trainer import *


def return_data( hbaromega_choose):    
    X = np.load("data/processed_extrapolation.npy", allow_pickle=True)
    ETr = X[()]["data"][:hbaromega_choose, 1:]
    h_omega = X[()]["data"][:hbaromega_choose, 0]
    N_Max = X[()]["Nmax"].reshape([-1])
    ts_ = np.concatenate([N_Max + j * 18 for j in range(repeat + 1)], axis=0)
    ts_ = np.vstack(ts_).astype(np.float32).reshape([-1, 1])

    print("X", ETr.shape, "N_Max", ts_.shape, "hbarOmega", h_omega.shape)

    #--------------------------------------------
    # Normalizing factors
    scale_gs = -32.5
    scale_ho = 50
    scale_Nmax = np.max(ts_)

    #--------------------------------------------
    # Normalize the dataset for efficiency
    ts_  = ts_/scale_Nmax
    x1 = (ETr/scale_gs)
    x2 = h_omega/scale_ho


    #--------------------------------------------
    # Reverse Normalization
    # print(np.max(x1), np.max(x2), scale_gs, scale_ho, scale_Nmax )
    # x1=x1*scale_gs
    # x2=x2*scale_ho
    # t=t*scale_Nmax
    x2= jnp.expand_dims( jnp.expand_dims(x2, 1), 1)
    x1= jnp.expand_dims( x1, 2)
    x2 = jnp.repeat(x2, x1.shape[1], axis=1)
    x = jnp.concatenate([x1, x2], axis =2)
    
    return ts_, x, scale_gs, scale_ho, scale_Nmax






def main(ts_, x, trainer, model, itrations, prints, factor, init_step):
    params, static =  eqx.partition(model, eqx.is_array)
    #-----------------------------------------------------------
    # initialize the loss function
    func = trainer.return_loss_grad
    #-----------------------------------------------------------
    ## initialize the optimizer
    optim=OptaxSolver(opt=optax.adamw(1e-04), fun=func, value_and_grad=True, has_aux=True,\
        jit=False)
      
    #-----------------------------------------------------------
    params = trainer.train__EUC__( (ts_, x, init_step, N_max_constraints, 1e-4),\
        params, static, optim, n_iter=itrations, print_iter=prints)
    model = eqx.combine(params, static)
    return trainer, model

#----------------------------------------------------------------
# The plotting
def generate_plot(filename, model, x):
    x0 = x[:,0,:]
    t=ts_.reshape([-1])
    xhat = jax.vmap(model, in_axes=(None, 0))(t, x0)
            
    #-----------------------------------------------
    import seaborn as sns
    sns.color_palette("bright")
    large = 20; med = 18; small = 16
    marker_size = 1.01
    lw = 0.1
    inten = 0.4
    def cm2inch(value):
        return value/2.54
    COLOR = 'darkslategray'
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(36),cm2inch(23.5)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'lines.markersize': marker_size,
            'ytick.labelsize': large,
            'figure.titlesize': large, 
                'text.color' : COLOR,
                'axes.labelcolor' : COLOR,
                'axes.linewidth' : 0.5,
                'xtick.color' : COLOR,
                'ytick.color' : COLOR}
    colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:cyan",
            "dodgerblue",
            "violet",
            "orangered",
            "maroon",
            "darkorange",
            "burlywood",
            "greenyellow",
            "tab:gray",
            "black",
            "rosybrown",
            "lightseagreen",
            "teal",
            "aqua",
            "darkolivegreen",
                ]

    plt.rcParams.update(params)
    plt.rc('text', usetex = False)
    plt.figure()
    [plt.plot(ts_[0:9,:]*scale_Nmax, x[i,:, 0]*scale_gs, linestyle='-', c=colors[i], label=x[i,0,1]*scale_ho) for i in range(5)]
    [plt.plot(ts_*scale_Nmax, xhat[i,:, 0]*scale_gs, linestyle='--',c=colors[i], label=x[i,0,1]*scale_ho) for i in range(5)]
    # plt.plot(t, x1hat)
    # plt.xlim([0,1])
    plt.ylim([-31.8,-32.4])
    plt.xlabel('NMax')
    plt.ylabel('E (Ground State)')
    plt.grid(linestyle=':', linewidth=0.5)
    plt.legend()
    plt.savefig(filename, dpi=1000)


########################################################################################
def load_checkpoint(path, device="cpu"):
    #-----------------------------------------------------------
    trainer = Trainer()
    key = jax.random.PRNGKey(SEED)
    data_key, model_key, loader_key = jr.split(key, 3)

    #-----------------------------------------------------------
    # Initialize the model and load weights from a stored model
    model = NeuralODE(data_size=2, width_size=128, depth=3, key=model_key)

    # optimizer = optim.QHAdam(model.parameters(), lr=1e-3)
    if os.path.exists(path):
        model= eqx.tree_deserialise_leaves(path, model)
    else:
        print("initialized model from scratch")
        eqx.tree_serialise_leaves(path, model)
    return trainer, model



## The main run loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_models.py",
        description="Tests the ODEVAE model for the No-Core Shell Model (NCSM)",
    )
    subparsers = parser.add_subparsers(help="", dest="command")

    parser.add_argument(
        "-cuda",
        "--cuda",
        default=True,
        action="store_true",
        help="consider using CUDA if it is available",
    )
    parser.add_argument(
        "-np",
        "--num-processes",
        default=None,
        help="number of processes to use (default is to auto detect)",
    )
    parser.add_argument(
        "-n",
        "--num-models",
        default=2,
        help="number of models in the ensemble",
    )

    train_parser = subparsers.add_parser("train")
    plot_parser = subparsers.add_parser("plot")
    list_parser = subparsers.add_parser("list")

    plot_parser.add_argument("models_path", help="directory where models are stored")
    list_parser.add_argument("models_path", help="directory where models are stored")
    list_parser.add_argument(
        "-e",
        "--epochs",
        default=25000,
        type=int,
        help="number of epochs that define a complete training",
    )
    list_parser.add_argument(
        "-i",
        "--incomplete",
        default=False,
        action="store_true",
        help="show models where training is incomplete only",
    )
    train_parser.add_argument("models_path", help="directory where models are stored")
    train_parser.add_argument(
        "-s", "--save", default=10, type=int, help="frequency of save"
    )
    train_parser.add_argument(
        "-e", "--epochs", default=25000, type=int, help="number of epochs to train for"
    )

    args = parser.parse_args()
    
    #--------- The Main Code ------------------------------
    #------------------------------------------------------
    #--------- The  parameters and everything ------------------------------
    jax.config.update("jax_enable_x64", False)
    N_Max_points      = 9
    SEED=5678
    N_max_constraints = 20
    hbaromega_choose  = 13
    repeat=5

    #-----------------------------------------------------------
    # Get data
    ts_, x, scale_gs, scale_ho, scale_Nmax=return_data( hbaromega_choose)

    if args.command == "list":
        model_path = "models/MLP__Extrapolation_vdist"+str(hbaromega_choose)+".eqx"
        trainer, model = load_checkpoint(model_path, device="cpu")    
        x= x.astype(jnp.float32)
        x0 = x[:,0,:]
        t=ts_.reshape([-1])
        config= {'int_step': 200,\
            'N_Max_constraints': N_max_constraints,\
            'dist_flag': 1e-04, 'step': 0}
        batch=(t, x0, x, config)
        params, static =  eqx.partition(model, eqx.is_array)
        print("The loss of the model is",\
            trainer.loss_fn_mse(params, static, batch, loss=True) )
            
    elif args.command == "train":
        model_path = "models/MLP__Extrapolation_vdist"+str(hbaromega_choose)+".eqx"
        trainer, model = load_checkpoint(model_path, device="cpu")   
        #---------------------------------------------------------------
        # If training
        trainer, model = main(ts_, x, trainer, model, itrations=400,\
            prints=100, factor=1e-04, init_step=10)
        
        #----------------------------------------------------------------
        # Save the model
        eqx.tree_serialise_leaves("models/MLP__Extrapolation_vdist"+str(hbaromega_choose)+".eqx", model)
        
        
    elif args.command == "plot":
        model_path = "models/MLP__Extrapolation_vdist"+str(hbaromega_choose)+".eqx"
        trainer, model = load_checkpoint(model_path, device="cpu")   
        #---------------------------------------------------------------
        generate_plot("Figures/plot_"+str(hbaromega_choose)+".png", model, x)


