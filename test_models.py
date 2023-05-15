import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable
import math
import os
import itertools

import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
sns.color_palette("bright")

def to_np(x):
    return x.detach().cpu().numpy()

def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.01
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z



## Let us now figure out how to get a model.
class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]
        out = self.forward(z, t)
        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True)
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten()
                              for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        # print("the data", t, t.size(), z0.size())
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]


class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2     = nn.Linear(hid_dim, hid_dim)
        self.lin3     = nn.Linear(hid_dim, in_dim)
        self.elu      = nn.ELU(inplace=True)

    def forward(self, x, t):
        # print(x.shape, t.shape)
        if not self.time_invariant:
            x = torch.cat((x, t.reshape([1,-1]) ), dim=-1)
        # print(x.shape)
        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out



class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.rnn = nn.GRU(input_dim+1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2*latent_dim)

    def forward(self, x, t):
        # Concatenate time to input
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        xt = torch.cat((x, t), dim=-1)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.hid2lat(h0[0])
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]
        return z0_mean, z0_log_var

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        func = NNODEF(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        zs = self.ode(z0, t, return_whole_sequence=True)

        hs = self.l2h(zs)
        xs = self.h2o(hs)
        return xs

class ODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(ODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = RNNEncoder(output_dim, hidden_dim, latent_dim)
        self.decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)

    def forward(self, x, t, MAP=False):
        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        x_p = self.decoder(z, t)
        return x_p, z, z_mean, z_log_var

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p


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
    prev_epoch, ode_trained, optimizer, prev_loss = step_model_optimizer_loss
    print("Starting training from epoch ", prev_epoch, flush=True)
    for i in range(prev_epoch, epochs):
        obs_, ts_, ho_ = create_batch_latent(ETr, h_omegaT, N_Max)
        input_d = torch.cat( [obs_, ho_], axis=2).to(device)
        x_p, z, z_mean, z_log_var = ode_trained(input_d, ts_.to(device))
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
        error_loss = 0.5 * ((input_d-x_p)**2).sum(-1).sum(0) / noise_std**2
        loss = torch.mean(error_loss+ kl_loss)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % print_iter == 0:
            print("(Print) Epoch:", i, "Total_Loss: "+str(loss.item()) + " with error", str(torch.mean(error_loss).item())+\
                  " KL divergence " + str(torch.mean(kl_loss).item()), flush=True)
        if i % save_iter == 0 or i == (epochs-1):
            print("(Save) Epoch:", i, "Total_Loss: "+str(loss.item()) + " with error", str(torch.mean(error_loss).item())+\
                  " KL divergence " + str(torch.mean(kl_loss).item()), flush=True)
            torch.save({
                'step': i,
                'model_state_dict': ode_trained.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                },
                save_path)
            obs_, ts_, ho_ = create_batch_latent_order(E, h_omega, N_Max)
            input_d = torch.cat([obs_, ho_], axis=2)
            samp_trajs_p = to_np(ode_trained.generate_with_seed(input_d, ts_))
            #print(samp_trajs_p.shape)

            plt.figure()
            fig, axes = plt.subplots(nrows=3, ncols=6, facecolor='white', figsize=(9, 9),\
                    gridspec_kw={'wspace': 0.5, 'hspace': 0.5}, dpi=400)
            axes = axes.flatten()
            for j, ax in enumerate(axes):
                ax.plot(18*ts_[:, j, 0], input_d[:, j, 0], label='real', linewidth = 1)
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
        PATH = 'models/Trained_ode_'+str(i)
        _, ode_trained, _, _ = load_checkpoint(PATH)
        samp_trajs_p = to_np(ode_trained.generate_with_seed(input_d, ts_))
        traj_mean.append( np.mean(samp_trajs_p, axis = 1) )
        traj_var.append(np.std(samp_trajs_p, axis=1)**2)

    plt.figure()
    traj_mean = np.array(traj_mean)
    traj_var = np.array(traj_var)
    mu = np.mean(traj_mean, axis = 0)
    var = np.mean(traj_var, axis = 0)
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
    plt.title("Ground State Energy averaged w.r.t. $\\overline{h} \\Omega$")
    plt.savefig('Figures/reconstruction_extrapolation_averaged_homega__.png', dpi=300,  bbox_inches="tight")
    plt.close()


def plot_model_averaged_(n_models, X):
    traj = []
    E = X[()]['data'][:, 1:]
    h_omega = X[()]['data'][:, 0]/50
    N_Max = X[()]['Nmax'].reshape([-1])/18
    obs_, ts_, ho_ = create_batch_latent_order(E, h_omega, N_Max)
    input_d = torch.cat([obs_, ho_], axis=2)
    for i in range(n_models):
        PATH = 'models/Trained_ode_'+str(i)
        step, ode_trained, _, _ = load_checkpoint(PATH)
        samp_trajs_p = to_np(ode_trained.generate_with_seed(input_d, ts_))
        traj.append(samp_trajs_p[:, :, 0])


    mu = np.mean(np.array(traj), axis=0)
    var = np.std(np.array(traj), axis=0)


    plt.figure()
    fig, axes = plt.subplots(nrows=3, ncols=6, facecolor='white', figsize=(
        16, 9),  gridspec_kw={'wspace': 0.5, 'hspace': 0.5}, dpi=400)
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        ax.scatter((18*ts_[:, 0, 0]).reshape([-1, 1]),  mu[:, j],
            label="predicted", marker='*', c=samp_trajs_p[:, j, 0], cmap=cm.plasma)
        ax.fill_between(18*ts_[:, 0, 0].reshape([-1]), (mu[:, j] - var[:, j] ).reshape([-1]),\
            (mu[:, j]+var[:, j]).reshape([-1]), color='gray', alpha=0.2)
        ax.grid("True")
        ax.set_xlabel('NMax, \n h$\\Omega$ = '+str(np.round(ho_[0, j, 0].item()*50, 2)), fontsize=10)
        if j == 5:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        if j == 0 or j == 6 or j == 12:
            ax.set_ylabel('Ground state Energy', fontsize=10)
            ax.set_title("Ground State Energy averaged  w.r.t. models")
        ax.set_ylim([-15, -33])
        # ax.set_xlim([-35,0])

    plt.savefig('Figures/reconstruction_extrapolation_model_averaged__.png',
                dpi=300,  bbox_inches="tight")
    plt.close()


def load_checkpoint(path):
    step = 0
    loss = 0
    model = ODEVAE(2, 128, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if os.path.exists(path):
        checkpoint = torch.load(PATH)
        step = checkpoint['step']
        loss = checkpoint['loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return step, model, optimizer, loss


def train_model(stuff):
    i, model, device = stuff
    PATH = 'models/Trained_ode_'+str(i)
    X = np.load('data/processed_extrapolation.npy', allow_pickle=True)
    print(F'Launch training of model {i} on process {mp.current_process().pid}', flush=True)
    conduct_experiment_latent(X, model, PATH, device=device, epochs=30000, save_iter=100, print_iter=10)


## The main run loop
if __name__ == '__main__':
    mp.set_start_method('spawn')
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        devices = [torch.device('cuda:%d'%i) for i in range(torch.cuda.device_count())]
        num_devices = len(devices)
        num_processes = num_devices
    else:
        num_processes = 1
        devices = [torch.device('cpu')] * num_processes
        num_devices = len(devices)
    print(F'use_cuda = {use_cuda}, num_devices = {num_devices}, num_processes = {num_processes}', flush=True)

    odes = []
    n_models__ = 100
    for i in range(n_models__):
        PATH = 'models/Trained_ode_'+str(i)
        checkpoint = load_checkpoint(PATH)
        checkpoint[1].to(devices[i%num_devices])
        odes.append(checkpoint)
    stuff = zip(range(len(odes)), odes, itertools.cycle(devices))
    with mp.Pool(num_processes) as pool:
        pool.map(train_model, stuff)

    X = np.load('data/processed_extrapolation.npy', allow_pickle=True)
    plot_homega_average_(n_models__, X)
    plot_model_averaged_(n_models__, X)

