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
    def loss_fn_mse(self, params, statics, batch,\
                    loss=False, n_points=9):
        # ---------------------------------------------
        t, x0, x, config = batch
        # ---------------------------------------------
        # init_step = config["int_step"]
        # N_max_constraints = config["N_Max
        # _constraints"]
        # # dist_flag = config["dist_flag"]
        # step = config["step"]
        x0 = jnp.asarray(x0)
        x  = jnp.asarray(x)
        
        
        # print(x0.shape, x.shape)
        # start=40
        # ---------------------------------------------
        model = eqx.combine(params, statics)
        xhat = jax.vmap(model, in_axes=(None, 0))(t, x0)
        # vect_mean = jax.lax.stop_gradient(jnp.mean(xhat.at[:, -1, 0].get(), axis = 0).item())
                # vect_var = jnp.std(xhat.at[:, -1, 0].get(), axis = 0).item()
        # -----------------------------------------------------------------
        # Go to the same point constraints
        # print(x.shape[1], t[0:x.shape[1]], t )
        vect_mean = 0.9165
        factor_E = ( 1 / (0.52-t[:n_points]) )
        Ehat = xhat[:, :n_points, 0]
        E = x[:, :n_points, 0]
        
        diff = factor_E*(Ehat-E)
        # error_last = jnp.sum( ((1/(0.17-t[x.shape[1]])*2)*(x[:,x.shape[1], :]-xhat[:, x.shape[1], :]))**2 )
        error = jnp.mean(diff**2)                
        # --------------------------------
        start=20      
        X1 = xhat[:, start:-1, 0]
        X2 = xhat[:, (start+1):, 0]
        diff = (X2 - vect_mean )**2      
        
        error_grad = jnp.sqrt(
                    jnp.sum(
                        diff
                    )
                )
        
        diff = jnp.sum(( vect_mean - xhat.at[:, -1, :].get())**2 ) 
        error_same_point = jnp.linalg.norm(diff)
        L  = error + 1e-2*(error_grad+0.1*error_same_point)        
        #+0.01*
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
            return L,  (L, error, error_same_point, error_grad, xhat)
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

    
    def save_opt_state_into_pkl(self, pkl_path, state):
        import cloudpickle
        epoch, opt_state, optimizer = state
        with open(pkl_path, "wb") as p:
            params = { 'epoch': epoch,
            'opt_state': opt_state, 
            'optimizer': optimizer
            }
            cloudpickle.dumps(params, p)
        print("saved things")
    
    # -------------------------------------------------------------
    def train__EUC__(self,\
        trainloader,
        params,
        static,
        optim,
        model_path,
        n_iter=1000,
        save_iter=200,
        print_iter=200,
        switch=1000,
        n_points=4
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
            # print(batch, params, static)
            grads, Losses = jax.grad(self.loss_fn_mse, has_aux=True)(params, static, batch, loss = True, n_points=n_points)
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
        # linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=1000, verbose=True)
        # optim = optax.chain(optim, linesearch)
        pbar = tqdm(range(init_step, n_iter), initial=init_step, total=n_iter)
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


            if step==switch:
                optim = optax.lbfgs(learning_rate = 1, memory_size=100)
                # linesearch=optax.scale_by_zoom_linesearch(
                # max_linesearch_steps=50, verbose=True)
                opt_state =optim.init(params)
                save_iter=10
                print_iter=10
                print("The optimizer is switched", optim, save_iter)

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

            
            
            # -------------------------------------- -------------------
            # optax first order
            losses, grad = return_loss_grad(params=params)  
            (L, (L, error, dist, error_grad, yhat)) = losses
            updates, opt_state = optim.update(
                grad, opt_state, params, value=L,\
                grad=grad, value_fn=return_loss_grad_second
            )
            
            
            
            
            params = optax.apply_updates(params, updates)            
            # print(yhat[:,-1,0])

            if error<1e-5:
                model = eqx.combine(params, static)
                eqx.tree_serialise_leaves(model_path, model) 
                
                plt.figure()
                [
                    plt.plot(t[0:n_points], x[i, :n_points, 0], linestyle="-", c=colors[i])
                    for i in range(x.shape[0])
                ]
                [
                    plt.plot(t, yhat[i, :, 0], linestyle="--", c=colors[i])
                    for i in range(x.shape[0])
                ]
                # plt.plot(t, x1hat)
                # plt.xlim([0, 1])
                # plt.title( str(np.mean(yhat[:,-1,0])) + '(' + str( jnp.abs(np.max(yhat[:,-1,0])- np.min(yhat[:,-1,0])) )  + ')' )
                # plt.ylim([-30,-35])
                # plt.xlim([8,20])
                plt.xlabel("NMax")
                plt.ylabel("E (Ground State)")
                plt.grid(linestyle=":", linewidth=0.5)
                plt.savefig("Figures/training/plot_"+str(step)+"model_num"+str(model_num)+"_nmax"+str(n_points)+"_.png", dpi=500)
                plt.close()


 
                return params
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
                # self.save_opt_state_into_pkl('load__optimize.pkl', (step, optim, optim ))
                # lr = lr*0.99
                # optim = optax.adam(lr)
                
                
            if step % print_iter == 0:
                # print(x.shape, t.shape, x0.shape)
                # print(t[:9], t[:9]*60)
                # print(x, yhat)
                # t__ = np.concatenate([t[0:9].reshape([-1])*18, t[9:].reshape([-1])*18*3], axis = 0)
                plt.figure()
                [
                    plt.plot(t[0:n_points], x[i, :n_points, 0], linestyle="-", c=colors[i])
                    for i in range(x.shape[0])
                ]
                [
                    plt.plot(t, yhat[i, :, 0], linestyle="--", c=colors[i])
                    for i in range(x.shape[0])
                ]
                # plt.plot(t, x1hat)
                # plt.xlim([0, 1])
                # plt.title( str(np.mean(yhat[:,-1,0])) + '(' + str( jnp.abs(np.max(yhat[:,-1,0])- np.min(yhat[:,-1,0])) )  + ')' )
                # plt.ylim([0.5, 0.95])
                # plt.xlim([8,20])
                plt.xlabel("NMax")
                plt.ylabel("E (Ground State)")
                plt.grid(linestyle=":", linewidth=0.5)
                plt.savefig("Figures/training/plot_"+str(step)+"model_num"+str(model_num)+"_nmax"+str(n_points)+"_.png", dpi=500)
                plt.close()

        return params
