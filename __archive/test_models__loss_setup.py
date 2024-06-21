import argparse
import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import itertools
sns.color_palette("bright")
from odevae import *
import copy

def to_np(x):
    return x.detach().cpu().numpy()


def create_batch_latent(X, y, N_Max):
    idx = [np.random.randint(0, X.shape[0]) for _ in range(20)]
    obs_ = torch.from_numpy(X[idx, :].astype(np.float32).T).unsqueeze(2)
    ts_ = np.vstack(N_Max).astype(np.float32).reshape([-1, 1])
    ts_ = torch.from_numpy(np.repeat(ts_, obs_.shape[1], axis=1)).unsqueeze(2)
    ho_ = torch.from_numpy(np.repeat(y[idx].astype(
        np.float32).reshape([1, -1]), obs_.shape[0], axis=0)).unsqueeze(2)
    return obs_, ts_, ho_


def create_batch_latent_order(X, y, N_Max):
    idx = [i for i in range(X.shape[0])]
    obs_ = torch.from_numpy(X[idx, :].astype(np.float32).T).unsqueeze(2)
    ts_ = np.vstack(N_Max).astype(np.float32).reshape([-1, 1])
    ts_ = torch.from_numpy(np.repeat(ts_, obs_.shape[1], axis=1)).unsqueeze(2)
    ho_ = torch.from_numpy(np.repeat(y[idx].astype(
        np.float32).reshape([1, -1]), obs_.shape[0], axis=0)).unsqueeze(2)
    return obs_, ts_, ho_


def conduct_experiment_latent(X, step_model_optimizer_loss, save_path, device='cpu', epochs=5000, save_iter=1000, print_iter=100):
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))
    E = X[()]['data'][:, 1:]
    h_omega = X[()]['data'][:, 0]/50
    N_Max = X[()]['Nmax'].reshape([-1])/18
    n_points = h_omega.shape[0]
    noise_std = 1
    permutation = [np.random.randint(0, n_points) for k in range(10)]
    ETr = E[permutation]
    h_omegaT = h_omega[permutation]
    # Train Neural ODE
    prev_epoch, ode_trained, optimizer_adam, prev_loss = step_model_optimizer_loss
    print("Starting training from epoch ", prev_epoch, flush=True)
    L__ =[]
    for i in range(prev_epoch, epochs ):
        obs_, ts_, ho_ = create_batch_latent(ETr, h_omegaT, N_Max)
        input_d = torch.cat( [obs_, ho_], axis=2).to(device)
        x_p, z, z_mean, z_log_var = ode_trained(input_d, ts_.to(device))
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
        error_loss = 0.5 * ((input_d-x_p)**2).sum(-1).sum(0) / noise_std**2
        loss = torch.mean(error_loss+ kl_loss)
        optimizer_adam.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_adam.step()
        L__.append(loss.detach().numpy())
        if i % print_iter == 0:
            print("(Print) Epoch:", i, "Total_Loss: "+str(loss.item()) + " with error", str(torch.mean(error_loss).item())+\
                  " KL divergence " + str(torch.mean(kl_loss).item()), flush=True)
            


        if i % save_iter == 0 or i == epochs:
            print("(Save) Epoch:", i, "Total_Loss: "+str(loss.item()) + " with error", str(torch.mean(error_loss).item())+\
                  " KL divergence " + str(torch.mean(kl_loss).item()), flush=True)
            torch.save({
                'step': i,
                'model_state_dict': ode_trained.state_dict(),
                'optimizer_state_dict': optimizer_adam.state_dict(),
                'loss': loss
                },
                save_path)
            obs_, ts_, ho_ = create_batch_latent_order(E, h_omega, N_Max)
            input_d = torch.cat([obs_, ho_], axis=2).to(device)
            samp_trajs_p = to_np(ode_trained.generate_with_seed(input_d, ts_.to(device)))
            #print(samp_trajs_p.shape)

            plt.figure()
            fig, axes = plt.subplots(nrows=3, ncols=6, facecolor='white', figsize=(9, 9),\
                    gridspec_kw={'wspace': 0.5, 'hspace': 0.5}, dpi=400)
            axes = axes.flatten()
            for j, ax in enumerate(axes):
                ax.plot(18*to_np(ts_[:, j, 0]), to_np(input_d[:, j, 0]), label='real', linewidth = 1)
                ax.scatter(18*ts_[:, j, 0], samp_trajs_p[:, j, 0], 3,
                           label="predicted" , marker='*',
                           c=samp_trajs_p[:, j, 0], cmap=cm.plasma)
                ax.grid("True")
                ax.set_xlabel('NMax, \n h$\\Omega$ ='\
                    +str( np.round(ho_[0,j,0].item()*50, 2 )), fontsize = 10)
                if j == 5:
                    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                if j == 0 or j==6 or j ==12:
                    ax.set_ylabel('Ground state Energy', fontsize = 10)
            plt.text(20, 75, "\n Total loss: "+str(np.round(loss.item(), 2)) + "\n with error: "\
                + str(np.round(torch.mean(error_loss).item(), 2))+" \n KL divergence: " + str(np.round(torch.mean(kl_loss).item(), 2)))
            plt.savefig('Figures/training/reconstruction_'+str(i)+'.png', dpi= 300,  bbox_inches="tight")
            plt.close()


    optimizer_BFGS = torch.optim.LBFGS(ode_trained.parameters(),                    
                                    history_size=200, 
                                    max_iter=10)
    def closure():
        x_p, z, z_mean, z_log_var = ode_trained(input_d, ts_.to(device))
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
        error_loss = ((input_d-x_p)**2).sum(-1).sum(0) / noise_std**2
        loss = torch.mean(error_loss+0.0001*kl_loss)
        optimizer_BFGS.zero_grad()
        loss.backward()
        return loss
    obs_, ts_, ho_ = create_batch_latent(ETr, h_omegaT, N_Max)
    input_d = torch.cat( [obs_, ho_], axis=2).to(device)
    copy_net = copy.deepcopy(ode_trained)
    for i in range( 100):
        optimizer_BFGS.step(closure)
        if torch.isnan(closure()):
            print("The loss is nan, I am copying things")
            ode_trained = copy.deepcopy(copy_net)
            optimizer_BFGS = torch.optim.LBFGS(ode_trained.parameters(), lr=1,
                    history_size =30, 
                    max_iter=10)
        else:
            copy_net = copy.deepcopy(ode_trained)
            L__.append(loss.detach().numpy())
        
        if i % 10 == 0:
            obs_, ts_, ho_ = create_batch_latent(ETr, h_omegaT, N_Max)
            input_d = torch.cat( [obs_, ho_], axis=2).to(device)
            x_p, z, z_mean, z_log_var = ode_trained(input_d, ts_.to(device))
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
            error_loss = torch.mean((input_d-x_p)**2)
            loss = torch.mean(error_loss+ 0.0001*kl_loss)
            print("(Print) Epoch:", i, "Total_Loss: "+str(loss.item()) + " with error", str(torch.mean(error_loss).item())+\
                " KL divergence " + str(torch.mean(kl_loss).item()), flush=True)
    

        if i % 10 == 0 or i == 100:
            obs_, ts_, ho_ = create_batch_latent(ETr, h_omegaT, N_Max)
            input_d = torch.cat( [obs_, ho_], axis=2).to(device)
            x_p, z, z_mean, z_log_var = ode_trained(input_d, ts_.to(device))
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
            error_loss = torch.mean((input_d-x_p)**2)
            loss = torch.mean(error_loss+ kl_loss)

            print("(Save) Epoch:", i, "Total_Loss: "+str(loss.item()) + " with error", str(torch.mean(error_loss).item())+\
                " KL divergence " + str(torch.mean(kl_loss).item()), flush=True)
            torch.save({
                'step': epochs+i,
                'model_state_dict': ode_trained.state_dict(),
                'optimizer_state_dict': optimizer_adam.state_dict(),
                'loss': loss
                },
                save_path)
            obs_, ts_, ho_ = create_batch_latent_order(E, h_omega, N_Max)
            input_d = torch.cat([obs_, ho_], axis=2).to(device)
            samp_trajs_p = to_np(ode_trained.generate_with_seed(input_d, ts_.to(device)))
            #print(samp_trajs_p.shape)

            plt.figure()
            fig, axes = plt.subplots(nrows=3, ncols=6, facecolor='white', figsize=(9, 9),\
                    gridspec_kw={'wspace': 0.5, 'hspace': 0.5}, dpi=400)
            axes = axes.flatten()
            for j, ax in enumerate(axes):
                ax.plot(18*to_np(ts_[:, j, 0]), to_np(input_d[:, j, 0]), label='real', linewidth = 1)
                ax.scatter(18*ts_[:, j, 0], samp_trajs_p[:, j, 0], 3,
                        label="predicted" , marker='*',
                        c=samp_trajs_p[:, j, 0], cmap=cm.plasma)
                ax.grid("True")
                ax.set_xlabel('NMax, \n h$\\Omega$ ='\
                    +str( np.round(ho_[0,j,0].item()*50, 2 )), fontsize = 10)
                if j == 5:
                    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                if j == 0 or j==6 or j ==12:
                    ax.set_ylabel('Ground state Energy', fontsize = 10)
            plt.text(20, 75, "\n Total loss: "+str(np.round(loss.item(), 2)) + "\n with error: "\
                + str(np.round(torch.mean(error_loss).item(), 2))+" \n KL divergence: " + str(np.round(torch.mean(kl_loss).item(), 2)))
            plt.savefig('Figures/training/reconstruction_'+str(i)+'.png', dpi= 300,  bbox_inches="tight")
            plt.close()


    # print(L__)
    # np.savetxt('loss_values.csv', L__, delimiter=',')




########################################################################

## Plot the first one
def plot_homega_average_(n_models, X):
    traj_mean = []
    traj_var = []
    E = X[()]['data'][:, 1:]
    h_omega = X[()]['data'][:, 0]/50
    N_Max = X[()]['Nmax'].reshape([-1])/18
    obs_, ts_, ho_ = create_batch_latent_order(E, h_omega, N_Max)
    input_d = torch.cat([obs_, ho_], axis=2)

    for i in range(n_models):
        path = 'models/Trained_ode_'+str(i)
        _, ode_trained, _, _ = load_checkpoint(path)
        samp_trajs_p = to_np(ode_trained.generate_with_seed(input_d, ts_))
        traj_mean.append( np.mean(samp_trajs_p, axis = 1) )
        traj_var.append(np.std(samp_trajs_p, axis=1)**2)

    traj_mean = np.array(traj_mean)
    traj_var = np.array(traj_var)

    mu = np.mean(traj_mean, axis = 0)
    var = np.mean(traj_var, axis = 0)

    plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=1, facecolor='white', figsize=(
    8, 3),  gridspec_kw={'wspace': 0.5, 'hspace': 0.5}, dpi=400)
    axes.scatter((18*ts_[:,0, 0]).reshape([-1,1]),  mu[:,0],  label="predicted",\
        marker='*', c=samp_trajs_p[:, 0, 0], cmap=cm.plasma)
    axes.fill_between(18*ts_[:, 0, 0].reshape([-1]), (mu[:, 0] - var[:, 0] ).reshape([-1]),\
        (mu[:, 0]+var[:, 0]).reshape([-1]), color='gray', alpha=0.2)
    axes.grid("True")
    axes.set_xlabel('NMax', fontsize=10)
    axes.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    axes.set_ylabel('Ground state Energy', fontsize=10)
    axes.set_ylim(-100, 0)
    axes.set_xlim(left=4)
    plt.title("Ground State Energy averaged w.r.t. $\\overline{h} \\Omega$")
    plt.savefig('Figures/reconstruction_extrapolation_averaged_homega__.png', dpi=300,  bbox_inches="tight")
    plt.close()


def plot_model_averaged_(n_models, X):
    traj = []
    E = X[()]['data'][:, 1:]
    h_omega = X[()]['data'][:, 0]/50
    N_Max = (X[()]['Nmax'].reshape([-1]))
    old_N_Max = (X[()]['Nmax'].reshape([-1]))
    factor=np.max(N_Max)
    for i in range(5000):
        N_Max = np.concatenate([ N_Max, i*factor+old_N_Max])
    factor=np.max(N_Max)
    N_Max = N_Max/factor
    print(factor)
    obs_, ts_, ho_ = create_batch_latent_order(E, h_omega, N_Max)
    input_d = torch.cat([obs_, ho_], axis=2)
   
    for i in range(n_models):
        path = 'models/Trained_ode_'+str(i)
        _, ode_trained, _, _ = load_checkpoint(path)
        samp_trajs_p = to_np(ode_trained.generate_with_seed(input_d, ts_))
        traj.append(samp_trajs_p[:, :, 0])

    mu = np.mean(np.array(traj), axis=0)
    var = np.std(np.array(traj), axis=0)

    plt.figure()
    num_cols = 4
    num_rows = int(np.ceil(len(h_omega[h_omega*50<31]) / num_cols)) # limit plotted h_omega to up 30MeV
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, facecolor='white', figsize=(
        16, 12),  gridspec_kw={'wspace': 0.5, 'hspace': 0.5}, dpi=400)
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        ax.scatter((factor*ts_[:, 0, 0]).reshape([-1, 1]),  mu[:, j],
            label="predicted", marker='*', c=samp_trajs_p[:, j, 0], cmap=cm.plasma)
        ax.fill_between(factor*ts_[:, 0, 0].reshape([-1]), (mu[:, j] - var[:, j] ).reshape([-1]),\
            (mu[:, j]+var[:, j]).reshape([-1]), color='gray', alpha=0.2)
        ax.axhline(y=-32.2, color='r')
        ax.annotate('-32.2', (6, -33))
        ax.grid("True")
        ax.set_xlabel('NMax, \n h$\\Omega$ = '+str(np.round(ho_[0, j, 0].item()*50, 2)), fontsize=10)
        if j % num_cols == 0:
            ax.set_ylabel('Ground state Energy', fontsize=10)
            ax.set_title("Ground State Energy averaged  w.r.t. models")
        ax.set_ylim(-32.5, -30)
        ax.set_xlim(left=4)
        ax.get_legend_handles_labels()
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.savefig('Figures/reconstruction_extrapolation_model_averaged__.png',
                dpi=300,  bbox_inches="tight")
    plt.close()


def load_checkpoint(path, device='cpu'):
    step = 0
    loss = 0
    model = ODEVAE(2, 128, 6)
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        step = checkpoint['step']
        loss = checkpoint['loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return step, model, optimizer, loss


def train_model(stuff):
    i, model, device = stuff
    path = 'models/Trained_ode_'+str(i)
    X = np.load('data/processed_extrapolation.npy', allow_pickle=True)
    print(F'Launch training of model {i} on process {mp.current_process().pid}', flush=True)
    conduct_experiment_latent(X, model, path, device=device, epochs=3000, save_iter=500, print_iter=1000)


########################################################################
## The main run loop
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='test_models.py',
                        description='Tests the ODEVAE model for the No-Core Shell Model (NCSM)')
    parser.add_argument('-t', '--train', default=False, action='store_true', help='do training')
    parser.add_argument('-p', '--plot', default=True, action='store_true', help='generate plots with test data')

    args = parser.parse_args()

    mp.set_start_method('spawn')
    use_cuda = torch.cuda.is_available()
    print(torch.cuda.device_count())
    
    # use_cuda = True
    if use_cuda:
        devices = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        num_devices = len(devices)
        num_processes = num_devices
    else:
        num_processes = 8
        devices = [torch.device('cpu')] * num_processes
        num_devices = len(devices)


    print(F'use_cuda = {use_cuda}, num_devices = {num_devices}, num_processes = {num_processes}', flush=True)


    odes = []
    n_models__ = 1

    if args.train:
        for i in range(n_models__):
            path = 'models/Trained_ode_'+str(i)
            checkpoint = load_checkpoint(path, device=devices[i % num_devices])
            odes.append(checkpoint)
        stuff = zip(range(len(odes)), odes, itertools.cycle(devices))
        with mp.Pool(num_processes) as pool:
            pool.map(train_model, stuff)

    if args.plot:
        X = np.load('data/processed_extrapolation.npy', allow_pickle=True)
        plot_homega_average_(n_models__, X)
        plot_model_averaged_(n_models__, X)
