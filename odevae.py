import torch
from torch import Tensor
from torch import nn
import sys
import numpy as np
from odesolver import ForwardEuler, RRK

# ode_solver = ode_solver_relax = ForwardEuler(h_max=0.01)
# method = "FE"; h_max = 0.01
method = "RK44"; h_max = 0.1
ode_solver = RRK(h_max=h_max, rkm=method, relaxation=False)
ode_solver_relax = RRK(h_max=h_max, rkm=method, relaxation=True)

## Let us now figure out how to get a model.
class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]
        out = self.forward(z, t)
        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,),
            (z, t) + tuple(self.parameters()),
            grad_outputs=(a),
            allow_unused=True,
            retain_graph=True,
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
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
    # Not sure if there is a better way to signal back
    # to the NeuralODE class (a NN module) that  the
    # relaxation failed, so call turnOffRelax().
    relax_failed = False

    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func, relax):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                if relax:
                    try:
                        # If relaxation method fails (usually with a "time step too large" error)
                        # then we fall back to regular RK.
                        z0 = ode_solver_relax.solve(z0, t[i_t, 0, :], t[i_t + 1, 0, :], func)
                    except:
                        ODEAdjoint.relax_failed = True
                        z0 = ode_solver.solve(z0, t[i_t, 0, :], t[i_t + 1, 0, :], func)
                else:
                    z0 = ode_solver.solve(z0, t[i_t, 0, :], t[i_t + 1, 0, :], func)
                z[i_t + 1] = z0

        ctx.func = func
        ctx.relax = relax
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        relax = ctx.relax
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
            z_i, a = (
                aug_z_i[:, :n_dim],
                aug_z_i[:, n_dim : 2 * n_dim],
            )  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(
                    z_i, t_i, grad_outputs=a
                )  # bs, *z_shape
                adfdz = (
                    adfdz.to(z_i)
                    if adfdz is not None
                    else torch.zeros(bs, *z_shape).to(z_i)
                )
                adfdp = (
                    adfdp.to(z_i)
                    if adfdp is not None
                    else torch.zeros(bs, n_params).to(z_i)
                )
                adfdt = (
                    adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)
                )

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

            for i_t in range(time_len - 1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(
                    torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1)
                )[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat(
                    (
                        z_i.view(bs, n_dim),
                        adj_z,
                        torch.zeros(bs, n_params).to(z),
                        adj_t[i_t],
                    ),
                    dim=-1,
                )

                # Solve augmented system backwards
                if relax:
                    aug_ans = ode_solver_relax.solve(
                        aug_z, t_i[0, :], t[i_t - 1, 0, :], augmented_dynamics
                    )
                else:
                    aug_ans = ode_solver.solve(
                        aug_z, t_i[0, :], t[i_t - 1, 0, :], augmented_dynamics
                    )

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim : 2 * n_dim]
                adj_p[:] += aug_ans[:, 2 * n_dim : 2 * n_dim + n_params]
                adj_t[i_t - 1] = aug_ans[:, 2 * n_dim + n_params :]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(
                torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1)
            )[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None, None


class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func
        self.relax = True

    def turnOffRelax(self):
        self.relax = False

    def turnOnRelax(self):
        self.relax = True

    def forward(self, z0, t=Tensor([0.0, 1.0]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func, self.relax)
        if ODEAdjoint.relax_failed:
            self.turnOffRelax()
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
            self.lin1 = nn.Linear(in_dim + 1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t):
        if not self.time_invariant:
            x = torch.cat((x, t.reshape([1, -1])), dim=-1)
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

        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x, t):
        # Concatenate time to input
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.0
        xt = torch.cat((x, t), dim=-1)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.hid2lat(h0[0])
        z0_mean = z0[:, : self.latent_dim]
        z0_log_var = z0[:, self.latent_dim :]
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

    def turnOffRelax(self):
        self.decoder.ode.turnOffRelax()

    def turnOnRelax(self):
        self.decoder.ode.turnOnRelax()

    def forward(self, x, t, MAP=False):
        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        x_p = self.decoder(z, t)
        return x_p, z, z_mean, z_log_var

    def infer(self, seed_x, t):
        self.turnOffRelax() # no relaxation when inferring
        return self.generate_with_seed(seed_x, t)

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p
