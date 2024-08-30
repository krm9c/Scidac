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
    # @eqx.filter_jit
    def loss_fn_mse(self, params, statics, batch, loss=False):
        # ---------------------------------------------
        t, x0, x, config = batch
        # ---------------------------------------------
        # init_step = config["int_step"]
        # N_max_constraints = config["N_Max_constraints"]
        # # dist_flag = config["dist_flag"]
        # step = config["step"]
        x0 = jnp.asarray(x0)
        x  = jnp.asarray(x)
        # start=40
        # ---------------------------------------------
        model = eqx.combine(params, statics)
        xhat = jax.vmap(model, in_axes=(None, 0))(t, x0)
        # vect_max = jnp.argmax( jnp.abs(xhat.at[:, -1, 0].get()), axis = 0 )
        # xhat = xhat.at[:, start:, :].set(xhat.at[vect_max, start:, :].get())
        # -----------------------------------------------------------------
        # Go to the same point constraints
        factor_E = (1/(0.17-t[:x.shape[1]])*2)
        R = xhat[:, :x.shape[1], :]
        error = jnp.sum(factor_E*jnp.mean(\
                jnp.sum( (x-R)**2, axis =2) , axis=0) )         
        start=20
        ts_del = (t[start:] - t[(start - 1):-1]).reshape([-1,1])
        X1 = xhat.at[:, start:, :].get()
        X2 = xhat.at[:, (start-1):53, :].get()
        diff = jnp.sum((X1-X2), axis=2)
        finite_diff = jnp.dot(diff, ts_del)        
        error_grad = jnp.sqrt(
                    jnp.sum(
                        (  finite_diff  ) ** 2
                    )
                )
        L  = error   # +0.1*error_grad        
        
        # if loss:
        #     # if step%10==0:
        #         # print(step, "before changing dist flag", config["dist_flag"])
        #         # config["dist_flag"]=config["dist_flag"]*10
        #         # print("changing dist flag", config["dist_flag"])
        #         # if config["dist_flag"] >1000:
        #         #     print("changing the weights.")
        #         #     config["dist_flag"]=1
        #     if step > init_step:
        #         return 100*(Entropy+10*dist_Entropy)
        #     else:
        #         return 100*Entropy
        # else:
        if loss==True:
            return L,  (L, error, error, error_grad, xhat)
        else:
            return L
        
    # --------------------------------------------------
    # --------------------------------------------------
    # @eqx.filter_jit
    def evaluate__(self, params, static, batch):
        model = eqx.combine(params, static)
        t, x0, _, _ = batch
        yhat = jax.vmap(model, in_axes=(None, 0))(t, x0)
        # vect_max = jnp.argmax( jnp.abs(yhat.at[:, -1, 0].get()), axis = 0 )
        # yhat=yhat.at[:, -1, 0].set(yhat[vect_max, -1, 0])
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
        import numpy as np
        import optax
        t, x, init_step, N_max_constraints, dist_flag, model_num = trainloader
        x = x
        x0 = x[:, 0, :]
        t = t.reshape([-1])
        config = {
            "int_step": init_step,
            "N_Max_constraints": N_max_constraints,
            "dist_flag": dist_flag,
            "step": init_step,
        }
        batch = (t, x0, x, config)
        
        def return_loss_grad(params):
            grads, Losses = jax.grad(self.loss_fn_mse, has_aux=True)(params, static, batch, loss = True)
            L, Entropy, dist_Entropy, error, xhat= Losses
            # self.loss_fn_mse(params, statics, batch, loss=False)
            return (L, (L, Entropy, dist_Entropy, error, xhat) ), grads
        
        def return_loss_grad_second(params):
            return self.loss_fn_mse(params, static, batch, loss=False)
        
        
        # Jax opt way
        # opt_state = optim.init_state(params)
        # lbfgs_scaler = optax.scale_by_lbfgs()
        # scaler = optax.normalize_by_update_norm()
        # scaler_state = scaler.init(params)
        # linesearch = optax.scale_by_backtracking_linesearch(
        #     max_backtracking_steps=100, store_grad=True
        # )
        # linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=100, verbose=True)
        # optim = optax.chain(optim, linesearch)
        # optim = optax.lbfgs(learning_rate = 1e-04, memory_size=100)
        # linesearch=optax.scale_by_zoom_linesearch(
        # max_linesearch_steps=50, verbose=True)
        
        pbar = tqdm(range(n_iter))
        # value_and_grad_fun = optax.value_and_grad_from_state(return_loss_grad_second)   
        opt_state =optim.init(params)
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


            # ---------------------------------------------------------
            # jaxopt way
            # params, opt_state = optim.update(
            #     params=params, state=opt_state, statics=static, batch=batch  )            
            # params, opt_state = optim.run(params, statics=static, batch=batch)
            
            
            # ---------------------------------------------------------
            # optax L-bfgs
            # L, grad = value_and_grad_fun(params, state=opt_state)
            
            # (L, (L, error, dist, error_grad, yhat)) =\
            #     self.loss_fn_mse(params, static, batch,\
            #     loss=True)

            
            
            # ---------------------------------------------------------
            # optax first order
            losses, grad = return_loss_grad(params=params)  
            (L, (L, error, dist, error_grad, yhat)) = losses
            updates, opt_state = optim.update(
                grad, opt_state, params, value=L,\
                grad=grad, value_fn=return_loss_grad_second
            )
            params = optax.apply_updates(params, updates)            
            # print(yhat[:,-1,0])


            # updates, opt_state = optim.update(
            #     grad, opt_state, params, value=L, grad=grad,\
            #     value_fn=self.return_loss_grad, statics=static, batch=batch
            # )            
            # updates, scaler_state = scaler.update(grad, scaler_state, params)
            # updates, opt_state = opt.pdate(grad, opt_state, params,\
            #         value=L, grad=grad, value_fn=return_loss_grad)
            # updates, opt_state = optim.update(grad, opt_state, params)
            
            # params = step((params, opt_state)) # optax.apply_update(params, updates)
            
            # params = optax.apply_updates(params, updates)
            # print('Objective function: ', f(params))
            # (error, dist, error_grad, yhat) = opt_state.aux
            
            pbar.set_postfix(
                {
                    \
                    "L":L, "MSE:": error,\
                    "Distance Loss": dist,\
                    "gradient": error_grad
                    \
                }
            )
            if step % save_iter == 0:
                model = eqx.combine(params, static)
                eqx.tree_serialise_leaves(model_path, model)

            if step % print_iter == 0:
                # print(x.shape, t.shape, x0.shape)
                # print(t[:9], t[:9]*60)
                plt.figure()
                [
                    plt.plot(t[0:9]*60, x[i, :, 0]*(-32.5), linestyle="-", c=colors[i])
                    for i in range(x.shape[0])
                ]
                [
                    plt.plot(t*60, yhat[i, :, 0]*(-32.5), linestyle="--", c=colors[i])
                    for i in range(x.shape[0])
                ]
                # plt.plot(t, x1hat)
                # plt.xlim([0, 1])
                plt.title( str(np.mean(yhat[:,-1,0])*-32.5) + '(' + str( np.std(yhat[:,-1,0])**2*(32.5) )  + ')' )
                plt.ylim([-31.5,-32.5])
                plt.xlabel("NMax")
                plt.ylabel("E (Ground State)")
                plt.grid(linestyle=":", linewidth=0.5)
                plt.savefig("Figures/training/plot_"+str(step)+"model_num"+str(model_num)+"_.png", dpi=500)
                plt.close()

        return params
