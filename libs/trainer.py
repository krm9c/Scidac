from torch.utils.tensorboard import SummaryWriter
import copy
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np_
import pickle
import matplotlib.pyplot as plt
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


# plt.style.use('seaborn-white')
COLOR = "darkslategray"
params = {
    "axes.titlesize": small,
    "legend.fontsize": small,
    "figure.figsize": (cm2inch(36), cm2inch(23.5)),
    "axes.labelsize": med,
    "axes.titlesize": small,
    "xtick.labelsize": small,
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


class Trainer(eqx.Module):
    writer: SummaryWriter
    loss: str
    problem: str
    metric: str
    dict: dict

    def __init__(self, logdir="runs", Loss="class", metric="class", problem="vectors"):
        self.writer = SummaryWriter(logdir)
        self.loss = Loss
        self.problem = problem
        self.metric = metric
        self.dict = {}

    # --------------------------------------------------
    @eqx.filter_jit
    def loss_fn_mse(self, params, statics, batch, loss=True):
        # ---------------------------------------------
        t, x0, x, config = batch

        # ---------------------------------------------
        init_step = config["int_step"]
        N_max_constraints = config["N_Max_constraints"]
        dist_flag = config["dist_flag"]
        step = config["step"]
        x0 = jnp.asarray(x0)
        x = jnp.asarray(x)

        # ---------------------------------------------
        model = eqx.combine(params, statics)
        xhat = jax.vmap(model, in_axes=(None, 0))(t, x0)

        # -----------------------------------------------------------------
        # Go to the same point constraints
        start =  40 # (N_max_constraints+10)
        # print(init_step)
        # -----------------------------------------------------------------
        # print("prediction shape", xhat.shape)
        vect_min = jnp.argmax( jnp.abs(xhat[:, -1, 0]), axis = 0 )
        dist = jnp.linalg.norm( t[start:] * (xhat[vect_min, start:, 0] - xhat[:, start:, 0]))+jnp.sum((1-xhat[vect_min, start:, 0])**2)
        error = jnp.sum( (1/(0.5-t[:9])**2)*  jnp.sum( jnp.sum( (x - xhat[:, :9, :] )**2, axis =2) , axis=0) ) 
                
        ts_del = t[start:] - t[(start - 1) : -1]
        error_grad = jnp.mean(
            (
                t[(start - 1) : -1])
                * jnp.sqrt(
                    jnp.sum(
                        ((xhat[:, start:, 0] - xhat[:, (start - 1) : -1, 0]) / ts_del) ** 2
                    )
                )
            )

        if loss:
            if step > init_step:
                return error + dist_flag * dist
            else:
                return error
        else:
            return error, dist, error_grad, xhat

    # --------------------------------------------------
    @eqx.filter_jit
    def return_loss_grad(self, params, static, batch):
        grads = jax.grad(self.loss_fn_mse)(params, static, batch)
        loss, dist, error_grad, yhat = self.loss_fn_mse(
            params, static, batch, loss=False
        )
        return (loss, (loss, dist, error_grad, yhat)), grads

    # --------------------------------------------------
    @eqx.filter_jit
    def evaluate__(self, params, static, batch):
        model = eqx.combine(params, static)
        t, x0, _, _ = batch
        yhat = jax.vmap(model, in_axes=(None, 0))(t, x0)
        return yhat

    # -------------------------------------------------------------
    def train__EUC__(
        self,
        trainloader,
        params,
        static,
        optim,
        model_path,
        n_iter=1000,
        save_iter=200,
        print_iter=200,
    ):
        from tqdm import tqdm

        t, x, init_step, N_max_constraints, dist_flag, model_num = trainloader
        x = x.astype(np_.float32)
        x0 = x[:, 0, :]
        t = t.reshape([-1])
        config = {
            "int_step": init_step,
            "N_Max_constraints": N_max_constraints,
            "dist_flag": dist_flag,
            "step": init_step,
        }
        batch = (t, x0, x, config)
        opt_state = optim.init_state(init_params=params, static=static, batch=batch)

        # --------------------------------------------------------
        pbar = tqdm(range(n_iter))
        for step in pbar:
            batch = (
                t,
                x0,
                x,
                {
                    "int_step": init_step,
                    "N_Max_constraints": N_max_constraints,
                    "dist_flag": dist_flag,
                    "step": step,
                },
            )
            params, opt_state = optim.update(
                params=params, state=opt_state, static=static, batch=batch
            )
            (error, dist, error_grad, yhat) = opt_state.aux
            pbar.set_postfix(
                {"MSE:": error, "Distance Loss": dist, "gradient": error_grad}
            )

            if step % save_iter == 0:
                model = eqx.combine(params, static)
                eqx.tree_serialise_leaves(model_path, model)

            if step % print_iter == 0:
                # print(x.shape, t.shape, x0.shape)
                plt.figure()
                [
                    plt.plot(t[0:9], x[i, :, 0], linestyle="-", c=colors[i])
                    for i in range(x.shape[0])
                ]
                [
                    plt.plot(t, yhat[i, :, 0], linestyle="--", c=colors[i])
                    for i in range(x.shape[0])
                ]
                # plt.plot(t, x1hat)
                plt.xlim([0, 1])
                # plt.ylim([-28,-33])
                plt.xlabel("NMax")
                plt.ylabel("E (Ground State)")
                plt.grid(linestyle=":", linewidth=0.5)
                plt.savefig("Figures/training/plot_"+str(step)+"model_num"+str(model_num)+"_.png", dpi=500)
                plt.close()

        return params
