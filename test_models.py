# -----------------------------------------------------
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
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu' )
import jax.tree_util as tree
import numpy as np
import jax.numpy as jnp
from jaxopt import OptaxSolver
import jax.random as jr
import optax
# -------------------------------------------------------
## Regular Linear layer in equinox
from libs.model import *
from libs.utils import *
from libs.trainer import *
from sklearn.preprocessing import MinMaxScaler
def return_data(hbaromega_choose, plot=False):
    # --------------------------------------------
    # Normalizing factors
    scale_gs = -35
    scale_ho = 50
    repeat=20
    offset = 0.5
    X =  np.load("data/processed_extrapolation.npy", allow_pickle=True)
    N_Max= np.array(X[()]["Nmax"], dtype=np.float64).reshape([-1, 1])   
    ts_ = np.concatenate([N_Max + j * 18.0 for j in range(repeat + 1)], axis=0)
    ts_ = np.vstack(ts_).reshape([-1,1])
    norm = 20+ts_
    ts_ = (ts_/norm)    
    unscaled = np.concatenate([N_Max + j * 18 for j in range(repeat + 1)], axis=0)   
    
    if plot:
        ETr = np.array(X[()]["data"][:, 1:], dtype=np.float64) 
        h_omega = np.array(X[()]["data"][:, 0], dtype=np.float64) 
        N_Max = np.array(X[()]["Nmax"].reshape([-1]), dtype=np.float64) 
    else:
        ETr = np.array(X[()]["data"][:hbaromega_choose, 1:], dtype=np.float64) 
        h_omega = np.array(X[()]["data"][:hbaromega_choose, 0], dtype=np.float64) 
    
        
    x1 = ETr/scale_gs
    x2 = h_omega/scale_ho 
    # print("the  data finally")
    # print(ts_, ts_.shape)
    # print(x1)
    # print(x2)
    # --------------------------------------------
    # Reverse Normalization
    # print(np.max(x1), np.max(x2), scale_gs, scale_ho, scale_Nmax, x1.shape, x2.shape, x3.shape, x3.dtype)
    # x1=x1*scale_gs
    # x2=x2*scale_ho
    # t=t*scale_Nmax
    # # repeat NMax across the zeroth dimension 8 times
    # x3 = jnp.expand_dims(x3, 0)
    # x3 = jnp.repeat(x3, x1.shape[0], axis=0)
    
    
    # repeat hbar omega across the second dimensions 9 times
    x2 = jnp.expand_dims(jnp.expand_dims(x2, 1), 1)
    x2 = jnp.repeat(x2, x1.shape[1], axis=1)
    # just expand the ground state
    x1 = jnp.expand_dims(x1, 2)
    x = jnp.concatenate([x1, x2], axis=2)
    
    # Check across one hbar omega different NMax and different ground state.
    #print(x[0,:,0], x[0,:,1], x[0,:,2])
    return ts_, x, scale_gs, scale_ho, unscaled

def main(
    ts_,
    x,
    trainer,
    model,
    model_path,
    model_num,
    iterations,
    factor,
    init_step,
    save_iter=200,
    print_iter=200,
    switch=5000,
    n_points =9
):
    params, static = eqx.partition(model, eqx.is_array)
    # -----------------------------------------------------------
    # initialize the loss function
    # -----------------------------------------------------------
    # initialize the optimizer
    # optim = OptaxSolver(
    #     opt=optax.nadamw(1e-05),\
    #     fun=trainer.return_loss_grad,\
    #     value_and_grad=True,\
    #     has_aux=True,\
    #     jit=False
    # )
    
    # import jaxopt
    # optim = jaxopt.NonlinearCG(fun=trainer.return_loss_grad,
    #     value_and_grad=True,\
    #     has_aux=True,\
    #     jit=False,\
    #     unroll=False
    #     # linesearch='backtracking',\
    #     # linesearch_init='current',\
    # )
    optim = optax.adam(1e-04)
    # optim__path = 'load__optimize.pkl'
    # if os.path.exists(optim__path):
    #     epoch, opt_state, optim = load_opt_state_from_pkl(optim__path)
    #     print(f"loaded model {optim__path}")

    # t = np.linspace(0, max(ts_), 100)
    # optim = optax.scale_by_lbfgs()
    # -----------------------------------------------------------
    params = trainer.train__EUC__(
        (ts_, x, init_step, N_max_constraints, factor, model_num),
        params,
        static,
        optim,
        model_path=model_path,
        n_iter=iterations,
        save_iter=save_iter,
        print_iter=print_iter,
        switch=switch,
        n_points=n_points
    )
    model = eqx.combine(params, static)
    return trainer, model


# ----------------------------------------------------------------
# The plotting
def generate_plot(filename, model_name, model_list, num_curves=8, n_points=7):
    print("plotting="+filename)
    
    # ---------------------------------------------------
    ts_, x, scale_gs, scale_ho, scale_ts__= return_data(8)
    x = x.astype(jnp.float64)
    t = ts_.reshape([-1])
    
    
    # ---------------------------------------------------
    import seaborn as sns
    sns.color_palette("bright")
    def cm2inch(value):
        return value / 2.54
    COLOR = "darkslategray"
    large = 14
    med = 12
    small = 10
    marker_size = 1.01
    params = {
        "axes.titlesize":  med,
        "legend.fontsize": med,
        "figure.figsize": (cm2inch(32), cm2inch(18)),
        "axes.labelsize": med,
        "axes.titlesize": large,
        "xtick.labelsize": med,
        "lines.markersize": marker_size,
        "ytick.labelsize": large,
        "figure.titlesize": large,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "axes.linewidth": 0.5,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
    }
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

    from scipy.ndimage import gaussian_filter1d
    plt.rcParams.update(params)
    plt.rc("text", usetex=False)
    plt.figure(dpi=1000)
    scale_gs=-35
    xhat=[]    
    for i,num in enumerate(model_list):
        model_path = model_name+str(num)+".eqx"
        _, model = load_checkpoint(model_path, device="cpu")
        x0 = x[:, 0, :]
        t = ts_.reshape([-1])
        # t = np.arange(0,5,0.001).reshape([-1])
        # print(t)
        # print(t.shape)    
        xhat.append(jax.vmap(model, in_axes=(None, 0))(t, x0))
    
    ts_ = t
    xhat = jnp.array(xhat)
    mean = jnp.mean(xhat, axis = 0)
    print(mean.shape, scale_ts__.shape)
    np.savetxt('data.csv', np.concatenate([ scale_ts__.reshape([1, -1]),\
                                            scale_gs*mean[:, :, 0].reshape([ 8, scale_ts__.shape[0]]) ],\
                                            axis =0), delimiter=',')
    print("xhat", x.shape, mean.shape)
    Err = jnp.abs(x-mean[:,0:9,:])
    print("NMax", ts_ * scale_Nmax)
    print(Err.shape)
    print("-------------------------------------------------------------------------")
    for i in range(10):
        print(x[i,0, 1]*scale_ho,  "mean",   mean[i,:, 0]*scale_gs)
        print(x[i,0, 1]*scale_ho,  "actual", x[i,:, 0]*scale_gs)
        print(Err[i, :,0])
    print("-------------------------------------------------------------------------")
    var  =  jnp.std( mean[:,-1, 0], axis = 0)**2
    [ print(Err[i, :,0]) for i in range(num_curves) ]
    # print(mean[:,53,0])
    
    [
        plt.plot(
            scale_ts__[0:n_points],
            x[i, :n_points, 0]*scale_gs,
            linestyle="--",
            c=colors[i],
            label= str(   round(x[i, 0, 1]*scale_ho,1)  ),
        )
        for i in range(1, num_curves)
    ]
    
    [ 
        plt.errorbar(
            scale_ts__,
            mean[i, :, 0]*scale_gs,
            yerr=var,
            barsabove=True,
            color=colors[i],
            ecolor=colors[i],
            label= str(  round(x[i, 0, 1]*scale_ho,1 ) )
        )
        for i in range(1, num_curves)
    ]
    
    
    # plt.title("--  True; -  Predicted;  | Uncertainty | \n"
    #         # +
    #         # '$E^{\\infty}_{GS}=$'+str(np.max(mean[:,-1,0], axis=0)*scale_gs)+\
    #         # ' ('+str( np.std(mean[:,-1, 0], axis=0) )+')'
    #         )
    
    # plt.xticks([0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 50, 70, 90] )
    # plt.yticks([-31, -31.25, -31.5, -31.75, -32.0, -32.05, -32.10, -32.15, -32.30, -33])
    plt.ylim([-31, -33])
    #plt.xscale('log')
    plt.xlabel("NMax")
    plt.ylabel("E (Ground State)")
    plt.grid(linestyle=":", linewidth=0.5)
    plt.legend(title="$\\bar{h}\Omega$", ncol=2, title_fontsize=med, loc='upper left', fancybox=True)
    # plt.show()
    plt.savefig(filename, dpi=100)
    plt.close()
    
    
# ----------------------------------------------------------------
def generate_plot_hbar_omega(filename, model_name, model_list, x, n_points=7):
    print("plotting="+filename)
    # -----------------------------------------------
    import seaborn as sns

    sns.color_palette("bright")
    large = 20
    med = 18
    small = 16
    marker_size = 1.01
    lw = 0.1
    inten = 0.4

    def cm2inch(value):
        return value / 2.54

    COLOR = "darkslategray"
    params = {
        "axes.titlesize":  med,
        "legend.fontsize": med,
        "figure.figsize": (cm2inch(36), cm2inch(40)),
        "axes.labelsize": med,
        "axes.titlesize": large,
        "xtick.labelsize": med,
        "lines.markersize": marker_size,
        "ytick.labelsize": large,
        "figure.titlesize": large,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "axes.linewidth": 0.5,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
    }
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
    plt.rc("text", usetex=False)
    plt.figure()
    xhat=[]
    for i,num in enumerate(model_list):
        model_path = model_name+str(num)+".eqx"
        _, model = load_checkpoint(model_path, device="cpu")
        x0 = x[:, 0, :]
        t = ts_.reshape([-1])
        xhat.append(jax.vmap(model, in_axes=(None, 0))(t, x0))
       
    xhat = jnp.array(xhat)
    print(xhat.shape) 
    
    mean = jnp.mean(xhat, axis = 0)
    var  = jnp.sqrt(jnp.std(xhat, axis = 0))

    [
        plt.plot(
            x[:, i, 1] * scale_ho,
            x[:, i, 0] * scale_gs,
            linestyle="--",
            c=colors[i],
            label= str(   ts_[i]*scale_Nmax  ),
        )
        for i in range(9)
    ]
    [
        plt.errorbar(
            x[:, i, 1] * scale_ho,
            mean[:, i,  0] * scale_gs,
            yerr=var[:,i, 0],
            barsabove=True,
            ecolor=colors[i],
            label= str( ts_[i]*scale_Nmax ),
        )
        for i in range(9)
    ]
    
    plt.title("--  True; -  Predicted;  | Uncertainty")
    # plt.ylim([-30, -33])
    # plt.xlim([5, 22.5])
    plt.xlabel("$\\bar{h}\Omega$ (MeV)" )
    plt.ylabel("E (Ground State)")
    plt.grid(linestyle=":", linewidth=0.5)
    plt.legend(title="$NMax$",ncol=2, title_fontsize=large, loc='lower right', fancybox=True)
    plt.savefig(filename, dpi=1000)
    plt.close()


########################################################################################
def load_opt_state_from_pkl(pkl_path):
    with open(pkl_path, "wb") as p:
        params = pickle.loads(p)    
    epoch = params['epoch']
    opt_state = params['opt_state']       
    optimizer= params['optimizer']
    return epoch, opt_state, optimizer 


########################################################################################
def load_checkpoint(path, device="cpu"):
    # -----------------------------------------------------------
    trainer = Trainer()
    # key = jax.random.PRNGKey(SEED)
    # data_key, model_key, loader_key = jr.split(key, 3)

    # -----------------------------------------------------------
    # Initialize the model and load weights from a stored model
    # model = NeuralODE(data_size=2, width_size=128, depth=3, key=model_key)
    
    model = NeuralODE(data_size=2, width_size=64, depth=3, key=SEED)
    if os.path.exists(path):
        print(f"loaded model {path}")
        model = eqx.tree_deserialise_leaves(path, model)
    else:
        print("initialized model from scratch")
        eqx.tree_serialise_leaves(path, model)
    return trainer, model

#----------------------------------------------------------------------------------
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
        "-m",
        "--model-num",
        default=0,
        help="which model",
    )


    #----------------------------------------------------------------------------------
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
    
    list_parser.add_argument(
        "-nmax",
        "--n_points",
        default=9,
        type=int,
        help="number of nmax points to be used in the training",
    )
    plot_parser.add_argument(
        "-nmax",
        "--n_points",
        default=9,
        type=int,
        help="number of nmax points to be used in the training",
    )
    
    
    train_parser.add_argument("models_path", help="directory where models are stored")
    train_parser.add_argument(
        "-s", "--save", default=200, type=int, help="frequency of save"
    )
    train_parser.add_argument(
        "-e", "--epochs", default=25000, type=int, help="number of epochs to train for"
    )
    train_parser.add_argument(
        "-os",
        "--optimizer_switch",
        default=5000,
        type=int,
        help="number of epochs at which to switch the optimizer",
    )
    
    train_parser.add_argument(
        "-nmax",
        "--n_points",
        default=9,
        type=int,
        help="number of nmax points to be used in the training",
    )
    
    
    args = parser.parse_args()
    
    #----------------------------------------------------------------------------------
    # --------- The Main Code -----------------------------------------------
    # --------- The  parameters and everything ------------------------------
    N_Max_points = 9
    SEED = 5678
    N_max_constraints = 20
    hbaromega_choose = 8
    repeat = 3
    init_step=1    
    dist_flag=1
    list_models=[0, 1, 2, 3]
    # print("The number of nmax points", args.n_points)
    # -----------------------------------------------------------
    # Get data
    ts_, x, scale_gs, scale_ho, scale_Nmax = return_data(hbaromega_choose)
    model_path = (
        f"models/MLP__Extrapolation_vdist{hbaromega_choose}_{args.n_points}_{args.model_num}.eqx"
    )
    
    # ----------------------------------------------------------------
    if args.command == "list":
        trainer, model = load_checkpoint(model_path, device="cpu")
        x = x.astype(jnp.float64)
        x0 = x[:, 0, :]
        t = ts_.reshape([-1])
        config = {
            "int_step": init_step,
            "N_Max_constraints": N_max_constraints,
            "dist_flag": dist_flag,
            "step": 0,
        }
        batch = (t, x0, x, config)
        loss_val = []
        
        for i,num in enumerate(list_models):
            model_name=f"models/MLP__Extrapolation_vdist{hbaromega_choose}_{args.n_points}_{num}.eqx"
            trainer, model = load_checkpoint(model_name, device="cpu")
            params, static = eqx.partition(model, eqx.is_array)
            loss_val.append( trainer.loss_fn_mse(params, static, batch) )
        
        loss_val=jnp.array(loss_val)    
        print(
            "The loss of the model is", jnp.mean(loss_val), jnp.sqrt(jnp.std(loss_val))
        )

    # ----------------------------------------------------------------
    elif args.command == "train":
        # ----------------------------------------------------------------
        trainer, model = load_checkpoint(model_path, device="cpu")
        trainer, model = main(
            ts_,
            x,
            trainer,
            model,
            model_path,
            model_num=args.model_num,
            iterations=args.epochs,
            save_iter=args.save,
            print_iter=args.save,
            factor=dist_flag,
            init_step=init_step,
            switch=args.optimizer_switch,
            n_points = args.n_points
        )
        
        # ----------------------------------------------------------------
        # Save the model
        eqx.tree_serialise_leaves(model_path, model)
        
        
    # ----------------------------------------------------------------
    elif args.command == "plot":
        # ts_, x, scale_gs, scale_ho, scale_Nmax = return_data(hbaromega_choose, plot=False)
        # x = x.astype(jnp.float64)
        # t = ts_.reshape([-1])
        model_path = (
        f"models/MLP__Extrapolation_vdist{hbaromega_choose}_{args.n_points}_{args.model_num}.eqx"
        )
        # ---------------------------------------------------------------
        generate_plot(f"Figures/plot_hbaromega{hbaromega_choose}_{args.n_points}.pdf",\
                    f"models/MLP__Extrapolation_vdist{hbaromega_choose}_{args.n_points}_",\
                    list_models)
        # generate_plot_hbar_omega(f"Figures/plot_with_respect_hbaromega{hbaromega_choose}.pdf",\
        #             f"models/MLP__Extrapolation_vdist{hbaromega_choose}_",\
        #             list_models)
