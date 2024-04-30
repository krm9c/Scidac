import argparse
import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
import os
import signal
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import itertools
import time
import torch_optimizer as optim
from odevae import *
sns.color_palette("bright")


def to_np(x):
    return x.detach().cpu().numpy()


def create_batch_latent(x, y, n_max):
    idx = [np.random.randint(0, x.shape[0]) for _ in range(20)]
    obs_ = torch.from_numpy(x[idx, :].astype(np.float32).T).unsqueeze(2)
    ts_ = np.vstack(n_max).astype(np.float32).reshape([-1, 1])
    ts_ = torch.from_numpy(np.repeat(ts_, obs_.shape[1], axis=1)).unsqueeze(2)
    ho_ = torch.from_numpy(
        np.repeat(y[idx].astype(np.float32).reshape([1, -1]), obs_.shape[0], axis=0)
    ).unsqueeze(2)
    return obs_, ts_, ho_


def create_batch_latent_order(x, y, N_Max, repeat=0):
    if repeat == 0:
        idx = [i for i in range(x.shape[0])]
        obs_ = torch.from_numpy(x[idx, :].astype(np.float32).T).unsqueeze(2)
        ts_ = np.vstack(N_Max).astype(np.float32).reshape([-1, 1])
        ts_ = torch.from_numpy(np.repeat(ts_, obs_.shape[1], axis=1)).unsqueeze(2)
        ho_ = torch.from_numpy(
            np.repeat(y[idx].astype(np.float32).reshape([1, -1]), obs_.shape[0], axis=0)
        ).unsqueeze(2)
        return obs_, ts_, ho_
    else:
        # print("The value of repeat", repeat)
        idx = [i for i in range(x.shape[0])]
        obs_ = torch.from_numpy(x[idx, :].astype(np.float32).T).unsqueeze(2)
        # print(N_Max.shape, obs_.shape)
        N_Max = np.concatenate([N_Max+j*18 for j in range(repeat+1) ], axis=0)
        scale = np.max(N_Max)
        N_Max = (N_Max/scale)
        # print("that N_Max", N_Max, N_Max.shape[0], obs_[8,:,:].shape)
        stackable = torch.vstack([obs_[(obs_.shape[0]-1),:,:].unsqueeze(0)
                     for i in range((N_Max.shape[0]-obs_.shape[0]))])        

        #########################################################################
        # print("stackable", obs_.shape, stackable.shape)
        obs_ext = torch.cat([obs_, stackable ], axis=0)
        ts_ = np.vstack(N_Max).astype(np.float32).reshape([-1, 1])
        ts_ = torch.from_numpy(np.repeat(ts_, obs_.shape[1], axis=1)).unsqueeze(2)
        ho_ = torch.from_numpy(np.repeat(y[idx].astype(np.float32).reshape([1, -1]), obs_ext.shape[0], axis=0)
        ).unsqueeze(2)

        
        # print(obs_.shape, obs_ext.shape, ts_.shape, ho_.shape)
        return obs_, obs_ext, ts_, ho_,  scale
        

import copy

def conduct_experiment_latent(
        
    X,
    step_model_optimizer_loss,
    save_path,
    device="mps",
    epochs=5000,
    save_iter=10000,
    print_iter=10000,
    plot_progress=True,
    repeat=5
):
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))
    ETr = X[()]["data"][:, 1:]
    h_omega = X[()]["data"][:, 0] / 50
    N_Max = X[()]["Nmax"].reshape([-1])
    n_points = h_omega.shape[0]
    noise_std = 1
    permutation = [np.random.randint(0, n_points) for k in range(10)]
    # Train Neural ODE
    prev_epoch, ode_trained, optimizer_adam, prev_loss = step_model_optimizer_loss
    print("Starting training from epoch ", prev_epoch, flush=True)
    begin_time = time.time()
    obs_, obs_ext, ts_, ho_, scale = create_batch_latent_order(ETr, h_omega, N_Max, repeat=repeat)
    input_d = torch.cat([obs_ext, ho_], axis=2).to(device)
    weights = copy.deepcopy(ts_[9:-1,0,0]).to(device)
    scheduler = 1e-03
    # optimizer_adam = torch.optim.LBFGS(ode_trained.parameters(),
    #                 history_size=100,
    #                 max_iter=10,
    #                 line_search_fn="strong_wolfe")
    
    print("prev epoch", prev_epoch, epochs)

    for i in range(prev_epoch, epochs):
        # if i > 1:
        #     ode_trained.turnOffRelax()
        # obs_, obs_ext, ts_, ho_, _ = create_batch_latent_order(ETr, h_omegaT, N_Max, repeat=repeat)
        def closure(flag=True):
            optimizer_adam.zero_grad()
            x_p, z, z_mean, z_log_var = ode_trained(input_d, ts_.to(device))
            kl_loss = -0.5 * torch.sum( 1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
            error_loss = 0.5*(((input_d[:9,:,:]- x_p[:9,:,:])**2)).sum(-1).sum(0) / noise_std ** 2

            ts_del = (ts_[10:,:,:]- ts_[9:-1,:,:]).to(device)
            error_grad = torch.sqrt(weights*torch.sum( torch.sum( ( (x_p[10:,:,:]- x_p[9:-1,:,:])/ts_del )**2, axis = 2) , axis = 1) )
            error_grad = torch.sum(error_grad)
            ##  calculate the cdistance for the last five
            dist_mat = []
            num_= 4
            for j in range(1,num_+1):
                # print(x_p.shape)
                # print("j is", j)
                vect = x_p[-(j+1), :, 0]

                # print("shape before squeeze", vect.shape)
                vect = vect.unsqueeze(dim =1)
                # print("shape after squeeze",vect.shape)
                dist__max = torch.cdist(vect, vect)
                # print(dist__max.shape)
                dist_mat.append(torch.linalg.norm(dist__max) )
            distance = torch.sum(torch.tensor(dist_mat).to(device))


            loss = torch.mean(error_loss+distance+scheduler*kl_loss)

            if flag:
                loss.backward()
                return loss
            else:
                return loss, error_loss, distance, error_grad, kl_loss
        
        optimizer_adam.step(closure)
        if i % save_iter == 0 or i == epochs:
            end_time = time.time()
            # Evaluation of the various terms in the loss function
            loss, error_loss, distance, error_grad, kl_loss = closure(flag=False)
        
            print(f"(Save)({(end_time-begin_time):.2f}s) Epoch: {i} Total Loss: {str(loss.item())} with error {str(torch.mean(error_loss).item())} with grad {str     (torch.sum(error_grad).item())}   KL divergence {str(torch.mean(kl_loss).item())} with distance {str(torch.sum(distance).item())}", flush=True)
            begin_time = time.time()
            torch.save(
                {
                    "step": i,
                    "model_state_dict": ode_trained.state_dict(),
                    "optimizer_state_dict": optimizer_adam.state_dict(),
                    "loss": loss,
                },
                save_path,
            )   
            
            if scheduler >1e-9:
                scheduler = scheduler*0.1
            else:
                scheduler = scheduler/(i+1)
                
            samp_trajs_p = to_np(
                ode_trained.generate_with_seed(input_d, ts_.to(device))
            )
            
            # print(samp_trajs_p.shape)
            if plot_progress:
                plt.figure()
                fig, axes = plt.subplots(
                    nrows=3,
                    ncols=6,
                    facecolor="white",
                    figsize=(9, 9),
                    gridspec_kw={"wspace": 0.5, "hspace": 0.5},
                    dpi=100,
                )
                axes = axes.flatten()
                for j, ax in enumerate(axes):
                    ax.scatter(
                        scale*N_Max,
                        to_np(obs_[:, j]),
                        6,
                        label="real",
                        marker="*",
                        c=obs_[:, j].cpu().numpy(),
                        cmap=cm.viridis,
                    )
                    ax.scatter(
                        scale * ts_[:, j, 0],
                        samp_trajs_p[:, j, 0],
                        3,
                        label="predicted",
                        marker="+",
                        c=samp_trajs_p[:, j, 0],
                        cmap=cm.plasma,
                    )
                    ax.grid("True")
                    ax.set_xlabel(
                        "NMax, \n h$\\Omega$ ="
                        + str(np.round(ho_[0, j, 0].item() * 50, 2)),
                        fontsize=10,
                    )
                    if j == 5:
                        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
                    if j == 0 or j == 6 or j == 12:
                        ax.set_ylabel("Ground state Energy", fontsize=10)
                    
                    ax.set_ylim([-32.3, -20])
                    ax.set_xlim(0,40)
                # plt.text(
                #     20,
                #     0,
                #     "\n Total loss: "
                #     + str(np.round(loss.item(), 2))
                #     + "\n with error: "
                #     + str(np.round(torch.mean(error_loss).item(), 2))
                #     + " \n KL divergence: "
                #     + str(np.round(torch.mean(kl_loss).item(), 2)) + " \n grad: "
                #     + str(np.round(torch.mean(error_grad).item(), 2)),
                # )
                fig_path ="Figures/"
                pathlib.Path(fig_path).mkdir(parents=True, exist_ok=True)
                plt.savefig("Figures/reconstruction_"+str(i)+str(device)+".png",
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close()

                
                
                #---------------------------------------------------------------------
                plt.figure()
                fig, ax = plt.subplots(
                    nrows=1,
                    ncols=1,
                    facecolor="white",
                    figsize=(9, 9),
                    gridspec_kw={"wspace": 0.5, "hspace": 0.5},
                    dpi=100,
                )
                axes = axes.flatten()
                for j in range(ts_.shape[1]):
                    # ax.scatter(
                    #     scale* to_np(ts_[:, j, 0]),
                    #     to_np(input_d[:, j, 0]),
                    #     6,
                    #     label="real",
                    #     marker="o",
                    #     c=input_d[:, j, 0].cpu().numpy(),
                    #     cmap=cm.viridis,
                    # )
                    ax.scatter(
                        scale * ts_[:, j, 0],
                        samp_trajs_p[:, j, 0],
                        3,
                        label="predicted",
                        marker="*",
                        c=samp_trajs_p[:, j, 0],
                        cmap=cm.plasma,
                    )
                
                ax.grid("True")
                ax.set_xlabel( "NMax")
                # if j == 5:
                # ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
                # if j == 0 or j == 6 or j == 12:
                ax.set_ylabel("Ground state Energy", fontsize=10)
                # ax.set_ylim([-32.3, -20])
                # ax.set_xlim(0,40)
                fig_path ="Figures/"
                pathlib.Path(fig_path).mkdir(parents=True, exist_ok=True)
                plt.savefig("Figures/reconstructionall_"+str(i)+str(device)+".png",
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close()




########################################################################
## Plot the first one
def plot_all(n_models, X, models_path, repeat):
    traj_mean = []
    traj_var = []
    E = X[()]["data"][:, 1:]
    h_omega = X[()]["data"][:, 0] / 50
    N_Max = X[()]["Nmax"].reshape([-1])
    obs_, obs_ext, ts_, ho_, scale = create_batch_latent_order(E, h_omega, N_Max, repeat=repeat)
    input_d = torch.cat([obs_ext, ho_], axis=2)
    
    traj=[]
    for i in n_models:
        path = f"{models_path}/Trained_ode_{str(i)}"
        _, ode_trained, _, _ = load_checkpoint(path)
        samp_trajs_p = to_np(ode_trained.infer(input_d, ts_))
        traj.append(samp_trajs_p[:, :, 0])
    mu = np.mean(np.array(traj), axis=0)
    var = np.std(np.array(traj), axis=0)
    
    
    #---------------------------------------------------------------------
    plt.figure()
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        facecolor="white",
        figsize=(9, 9),
        gridspec_kw={"wspace": 0.5, "hspace": 0.5},
        dpi=100,
    )
    # ax = ax.flatten()
    for j in range(ts_.shape[1]):
        ax.scatter(
            N_Max,
            to_np(obs_[:, j]),
            6,
            label="real",
            marker="o",
            c=obs_[:, j].cpu().numpy(),
            cmap=cm.viridis,
        )
        # ax.scatter(
        #     samp_trajs_p[:, j, 0],
        #     3,
        #     label="predicted",
        #     marker="*",
        #     c=samp_trajs_p[:, j, 0],
        #     cmap=cm.plasma,
        # )
        ax.errorbar(scale * ts_[:, j, 0], mu[:,j], yerr=np.sqrt(var[:,j]), ecolor='tab:blue',fmt='-',elinewidth=0.6, barsabove=True)
        
    ax.grid("True")
    ax.set_xlabel( "NMax")
    # if j == 5:
    # ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    # if j == 0 or j == 6 or j == 12:
    ax.set_ylabel("Ground state Energy", fontsize=10)
    # ax.set_ylim([-32.3, -20])
    # ax.set_xlim(0,40)
    fig_path ="Figures/"
    pathlib.Path(fig_path).mkdir(parents=True, exist_ok=True)
    plt.savefig("Figures/reconstructionall_.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
                
                
def plot_homega_average_(n_models, X, models_path, repeat):
    traj_mean = []
    traj_var = []
    E = X[()]["data"][:, 1:]
    h_omega = X[()]["data"][:, 0] / 50
    N_Max = X[()]["Nmax"].reshape([-1])
    obs_, obs_ext, ts_, ho_, ax_scale = create_batch_latent_order(E, h_omega, N_Max, repeat=repeat)
    print(ts_.shape, ho_.shape)
    input_d = torch.cat([obs_ext, ho_], axis=2)
    for i in n_models:
        path = f"{models_path}/Trained_ode_{str(i)}"
        _, ode_trained, _, _ = load_checkpoint(path)
        samp_trajs_p = to_np(ode_trained.infer(input_d, ts_))
        traj_mean.append(np.mean(samp_trajs_p, axis=1))
        traj_var.append(np.std(samp_trajs_p, axis=1) ** 2)
    traj_mean = np.array(traj_mean)
    traj_var = np.array(traj_var)
    mu = np.mean(traj_mean, axis=0)
    var = np.mean(traj_var, axis=0)

    plt.figure()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        facecolor="white",
        figsize=(8, 3),
        gridspec_kw={"wspace": 0.5, "hspace": 0.5},
        dpi=400,
    )
    axes.scatter(
        (ax_scale * ts_[:, 0, 0]).reshape([-1, 1]),
        mu[:, 0],
        label="predicted",
        marker="*",
        c=samp_trajs_p[:, 0, 0],
        cmap=cm.plasma,
    )
    axes.scatter(
            (ax_scale * ts_[:, 0, 0]).reshape([-1, 1]),
            np.mean(input_d[:, :, 0].numpy(), axis=1) ,
            label="original", marker="o",
            c=np.mean(input_d[:, :, 0].numpy(), axis=1),
            cmap=cm.viridis,
    )
    axes.fill_between(
        ax_scale * ts_[:, 0, 0].reshape([-1]),
        (mu[:, 0] - var[:, 0]).reshape([-1]),
        (mu[:, 0] + var[:, 0]).reshape([-1]),
        color="gray",
        alpha=0.2,
    )

    axes.grid("True")
    axes.set_xlabel("NMax", fontsize=10)
    axes.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    axes.set_ylabel("Ground state Energy", fontsize=10)
    axes.set_ylim(-33.5, -20)
    axes.set_xlim(0,30)
    plt.title("Ground State Energy averaged w.r.t. $\\overline{h} \\Omega$")
    plt.savefig(
        "Figures/reconstruction_extrapolation_averaged_homega__.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_model_averaged_(n_models, X, models_path, repeat=4):
    traj = []
    E = X[()]["data"][:, 1:]
    h_omega = X[()]["data"][:, 0] / 50
    N_Max = (X[()]["Nmax"].reshape([-1]))
    obs_,obs_ext, ts_, ho_, ax_scale = create_batch_latent_order(E, h_omega, N_Max, repeat=repeat)
    input_d = torch.cat([obs_ext, ho_], axis=2)
    print(obs_.shape, ho_.shape, input_d.shape)
    for i in n_models:
        path = f"{models_path}/Trained_ode_{str(i)}"
        _, ode_trained, _, _ = load_checkpoint(path)
        samp_trajs_p = to_np(ode_trained.infer(input_d, ts_))
        traj.append(samp_trajs_p[:, :, 0])


    mu = np.mean(np.array(traj), axis=0)
    var = np.std(np.array(traj), axis=0)

    plt.figure()
    num_cols = 4
    num_rows = int(
        np.ceil(len(h_omega[h_omega * 50 < 31]) / num_cols)
    )  # limit plotted h_omega to up 30MeV
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        facecolor="white",
        figsize=(16, 12),
        gridspec_kw={"wspace": 0.5, "hspace": 0.5},
        dpi=400,
    )

    # print("check before plotting", mu[:, 0].shape, input_d[:, 0, 0].shape)
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        # ax.scatter(
        #     (ax_scale * ts_[:, 0, 0]).reshape([-1, 1]),
        #     mu[:, j],
        #     label="predicted",
        #     marker="*",
        #     c=samp_trajs_p[:, j, 0],
        #     cmap=cm.plasma,
        # )

        ax.scatter(
            (N_Max).reshape([-1, 1]),
            obs_[:, j].numpy(),
            label="original",
            marker="o",
            c=obs_[:, j].numpy(),
            cmap=cm.viridis,
        )
        ax.errorbar(ax_scale * ts_[:, j, 0], mu[:,j], yerr=var[:,j], ecolor='tab:blue',fmt='-',elinewidth=0.6, barsabove=True)
                
        # ax.fill_between(
        #     (ax_scale * ts_[:, 0, 0]).reshape([-1]),
        #     (mu[:, j] - var[:, j]).reshape([-1]),
        #     (mu[:, j] + var[:, j]).reshape([-1]),
        #     color="gray",
        #     alpha=0.2,
        # )
        txt = '{0:.5}'.format(mu[-1,j])+'({0:.5})'.format( var[-1,j] )
        ax.text(x=50, y=-30, s=txt)
        ax.axhline(y=-32.2, color="r")
        ax.annotate("-32.2", (6, -33))
        ax.grid("True")
        ax.set_xlabel(
            "NMax, \n h$\\Omega$ = " + str(np.round(ho_[0, j, 0].item() * 50, 2)),
            fontsize=10,
        )
        if j % num_cols == 0:
            ax.set_ylabel("Ground state Energy", fontsize=10)
            ax.set_title("Ground State Energy averaged  w.r.t. models")
        ax.set_ylim(-32.3, -28.5)
        # ax.set_xlim(0,30)
        ax.get_legend_handles_labels()
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    plt.savefig(
        "Figures/reconstruction_extrapolation_model_averaged__.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_model_averaged_long_N_Max(n_models, X, models_path, repeat=4):
    traj = []
    E = X[()]["data"][:, 1:]
    h_omega = X[()]["data"][:, 0] / 50
    N_Max = (X[()]["Nmax"].reshape([-1]))
    _, obs_, ts_, ho_, ax_scale= create_batch_latent_order(E, h_omega, N_Max, repeat=repeat)
    print("what comes out", obs_.shape, ts_.shape, ho_.shape)
    input_d = torch.cat([obs_, ho_], axis=2)
    print(obs_.shape, ho_.shape, input_d.shape)
    for i in n_models:
        path = f"{models_path}/Trained_ode_{str(i)}"
        _, ode_trained, _, _ = load_checkpoint(path)
        samp_trajs_p = to_np(ode_trained.infer(input_d, ts_))
        traj.append(samp_trajs_p[:, :, 0])


    mu = np.mean(np.array(traj), axis=0)
    var = np.std(np.array(traj), axis=0)

    plt.figure()
    num_cols = 4
    num_rows = int(
        np.ceil(len(h_omega[h_omega * 50 < 31]) / num_cols)
    )  # limit plotted h_omega to up 30MeV
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        facecolor="white",
        figsize=(16, 12),
        gridspec_kw={"wspace": 0.5, "hspace": 0.5},
        dpi=400,
    )

    # print("check before plotting", mu[:, 0].shape, input_d[:, 0, 0].shape)
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        ax.scatter(
            (ax_scale * ts_[:,0,0]).reshape([-1, 1]),
            mu[:, j],
            label="predicted",
            marker="*",
            c=samp_trajs_p[:, j, 0],
            cmap=cm.plasma,
        )

        ax.scatter(
            (ax_scale * ts_[:,0,0]).reshape([-1, 1]),
            input_d[:, j, 0].numpy(),
            label="original",
            marker="o",
            c=input_d[:, j, 0].numpy(),

            cmap=cm.viridis,
        )

        ax.errorbar(
            (ax_scale * ts_[:, 0, 0]).reshape([-1]), mu[:, j].reshape([-1]), var[:, j].reshape([-1]),
            color="gray",
            alpha=0.4,
        )

        ax.axhline(y=-32.2, color="r")
        ax.annotate("-32.2", (6, -33))
        ax.grid("True")
        ax.set_xlabel(
            "NMax, \n h$\\Omega$ = " + str(np.round(ho_[0, j, 0].item() * 50, 2)),
            fontsize=10,
        )
        if j % num_cols == 0:
            ax.set_ylabel("Ground state Energy", fontsize=10)
            ax.set_title("Ground State Energy averaged  w.r.t. models")
        ax.set_ylim(-32.3, -20)
        ax.set_xlim(0,50)
        ax.get_legend_handles_labels()
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    plt.savefig(
        "Figures/reconstruction_extrapolation_model_averaged__long_NMax.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()





def load_checkpoint(path, device="cpu"):
    step = 0
    loss = 0
    model = ODEVAE(2,  512, 10)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = optim.QHAdam(model.parameters(), lr=1e-04)
    print("Initialized without checkpointing")
    
    if os.path.exists(path):
        print("load from checkpoint")
        checkpoint = torch.load(path,  map_location=device)
        step = checkpoint["step"]
        loss = checkpoint["loss"]
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        torch.save(
                {
                    "step": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                path,
            )   
            
            
    
        
    return step, model, optimizer, loss


def train_model(stuff):
    print("here in train_model")
    i, model, device, model_path, epochs, save_it = stuff
    print("Starting to train")
    # Register a handler for SIGTERM which removes the lock on the model.
    def sigterm_handler(*args):
        unlock_model(model_path)
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)

    if is_locked(model_path):
        return

    X = np.load("data/processed_extrapolation.npy", allow_pickle=True)
    print(
        f"Launch training of model {i} on process {mp.current_process().pid}",
        flush=True,
    )

    # Create a lock for this model so only this process/job can work on this model
    lock_model(model_path)
    
    conduct_experiment_latent(
        X, model, model_path, device=device, epochs=epochs, save_iter=save_it
        , print_iter=save_it
    )

    unlock_model(model_path)


########################################################################
# We want to be able to launch multiple jobs to train subsets of the
# ensemble of models concurrently. As such, we use a filesystem based
# semaphore to ensure a model is not being worked on by more than one job.
def lock_model(model_path):
    lock_file = open(model_path + ".lock", "w")
    lock_file.write(f"{mp.current_process().pid}")
    lock_file.close()


def unlock_model(model_path):
    os.remove(model_path + ".lock")


def is_locked(model_path):
    return os.path.isfile(model_path + ".lock")


########################################################################

## The main run loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_models.py",
        description="Tests the ODEVAE model for the No-Core Shell Model (NCSM)",
    )
    subparsers = parser.add_subparsers(help='', dest='command')

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

    train_parser = subparsers.add_parser("train")
    plot_parser = subparsers.add_parser("plot")
    list_parser = subparsers.add_parser("list")
    
    plot_parser.add_argument("models_path", help="directory where models are stored")
    list_parser.add_argument("models_path", help="directory where models are stored")
    list_parser.add_argument("-e", "--epochs", default=25000, type=int, help="number of epochs that define a complete training")
    list_parser.add_argument("-i", "--incomplete", default=False, action="store_true", help="show models where training is incomplete only")
    train_parser.add_argument("models_path", help="directory where models are stored")
    train_parser.add_argument("-s", "--save", default=10, type=int, help="frequency of save")
    train_parser.add_argument("-e", "--epochs", default=25000, type=int, help="number of epochs to train for")


    args = parser.parse_args()
    use_cuda = args.cuda
    if use_cuda:
        use_cuda = torch.cuda.is_available()
    if use_cuda:
        devices = [
            torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())
        ]
        num_devices = len(devices)
        num_processes = num_devices
        if args.num_processes is not None:
            num_processes = int(args.num_processes)
    else:
        # if torch.backends.mps.is_available():
        #     num_processes = 1
        #     devices = torch.device("mps")
        #     num_devices = 1
        # else:
        num_processes = 4
        if args.num_processes is not None:
            num_processes = int(args.num_processes)
        devices = [torch.device("cpu")] * num_processes
        num_devices = len(devices)

    print(
        f"use_cuda = {use_cuda}, num_devices = {num_devices}, num_processes = {num_processes}",
        flush=True,
    )

    odes = []
    model_paths = []
    n_models__ = [0,1]

    mp.set_start_method("spawn")
    if args.command == 'list':
        for i in n_models__:
            model_path = f"{args.models_path}/Trained_ode_{str(i)}"
            step, _, _, loss = load_checkpoint(model_path, device=devices[i % num_devices])
            if step < args.epochs or (not args.incomplete):
                print(f"Trained_ode_{str(i)}: step={step}, loss={loss}")

                
    elif args.command == 'train':
        device_list = []
        for i in n_models__:
            model_path = f"{args.models_path}/Trained_ode_{str(i)}"
            model_paths.append(model_path)
            checkpoint = load_checkpoint(model_path, device=devices[i % num_devices])
            odes.append(checkpoint)
            device_list.append(devices[i % num_devices])
            
        
                
        stuff = zip(range(len(odes)), odes, device_list, model_paths, itertools.repeat(args.epochs), itertools.repeat(args.save))
        with mp.Pool(num_processes) as pool:
            pool.map(train_model, stuff)


    elif args.command == 'plot':
        X = np.load("data/processed_extrapolation.npy", allow_pickle=True)
        repeat = 5
        plot_all(n_models__, X, args.models_path, repeat=repeat)
        # plot_homega_average_(n_models__, X, args.models_path, repeat=repeat)
        plot_model_averaged_(n_models__, X, args.models_path, repeat=repeat)
        # plot_model_averaged_long_N_Max(n_models__, X, args.models_path, repeat = repeat)