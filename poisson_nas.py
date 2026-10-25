"""Benchmark de POISSON 2D (tanh): mismas piezas que advection_nas.
Importar como:
    import poisson_nas as POI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import random


# Diagnóstico: concentración del softmax al final de la búsqueda
def softmax_concentration(supernet):
    """pi_max por capa buscable: cuánto 'pesa' la rama ganadora al acabar la
    búsqueda. 1.0 = one-hot (herencia exacta). 0.2 = uniforme con 5 ramas."""
    out = []
    with torch.no_grad():
        for i, a in enumerate(supernet.alpha_parameters):
            pi = torch.softmax(a, dim=0)
            k = int(torch.argmax(pi))
            out.append((i, supernet.op_names_per_layer[i][k], round(float(pi[k]), 3)))
    print("pi_max por capa:", out, "| min:", min(p for _, _, p in out))
    return out

# Semilla
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Decidimos función de activación
def _apply_activation(x, act_name):
    if act_name == 'tanh':
        return torch.tanh(x)
    elif act_name == 'relu':
        return F.relu(x)
    else:
        raise ValueError(f"Activación desconocida: {act_name}")

def _activation_module(act_name):
    if act_name == 'tanh':
        return nn.Tanh()
    elif act_name == 'relu':
        return nn.ReLU()
    else:
        raise ValueError(f"Activación desconocida: {act_name}")

# Capas mixtas
class MixedOperation(nn.Module):
    def __init__(self, input_size, output_sizes, max_width, activation='tanh'):
        super().__init__()
        self.output_sizes = list(output_sizes)
        self.max_width = max_width
        self.activation = activation
        self.op_names = [f'linear_{s}' for s in self.output_sizes] + ['skip']
        self.ops = nn.ModuleList()
        for out_size in self.output_sizes:
            layer = nn.Linear(input_size, out_size)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.ops.append(layer)
        self.ops.append(nn.Identity())

    def forward(self, x, alphas):
        out = 0.0
        for idx, op in enumerate(self.ops):
            op_out = op(x)
            if isinstance(op, nn.Linear):
                op_out = _apply_activation(op_out, self.activation)
            if op_out.size(1) < self.max_width:
                pad = torch.zeros(op_out.size(0), self.max_width - op_out.size(1),
                                  device=op_out.device, dtype=op_out.dtype)
                op_out = torch.cat([op_out, pad], dim=1)
            out = out + alphas[idx] * op_out
        return out

# Superred
class PINNDARTSNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, max_layers=8, neuron_options=[10,20,30,40], activation='tanh'):
        super().__init__()
        self.max_layers = max_layers
        self.neuron_options = list(neuron_options)
        self.max_width = max(self.neuron_options)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.input_layer = nn.Linear(input_dim, self.max_width)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        self.mixed_layers = nn.ModuleList()
        for _ in range(max_layers - 1):
            self.mixed_layers.append(MixedOperation(self.max_width, self.neuron_options, self.max_width, activation=self.activation))

        self.output_layer = nn.Linear(self.max_width, output_dim)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        self.alpha_parameters = nn.ParameterList()
        self.op_names_per_layer = []
        num_ops = len(self.neuron_options)+ 1
        for _ in range(max_layers - 1):
            p = nn.Parameter(torch.randn(num_ops)*1e-3)
            self.alpha_parameters.append(p)
            names = [f'linear_{s}' for s in self.neuron_options]+ ['skip']
            self.op_names_per_layer.append(names)

    def forward(self, x, discrete_architecture=False):
        x = _apply_activation(self.input_layer(x), self.activation)
        for i, mixed_layer in enumerate(self.mixed_layers):
            if discrete_architecture:
                raw = self.alpha_parameters[i]
                idx_max = torch.argmax(raw).item()
                op = mixed_layer.ops[idx_max]
                x = op(x)
                if isinstance(op, nn.Linear):
                    x = _apply_activation(x, self.activation)
                if x.size(1) < self.max_width:
                    pad = torch.zeros(x.size(0), self.max_width - x.size(1),
                                      device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=1)
            else:
                alphas = torch.softmax(self.alpha_parameters[i], dim=0)
                x = mixed_layer(x, alphas)
        return self.output_layer(x)

    def get_discrete_architecture(self):
        arch = []
        for i in range(len(self.alpha_parameters)):
            raw = self.alpha_parameters[i]
            idx = int(torch.argmax(raw).item())
            name = self.op_names_per_layer[i][idx]
            if 'linear' in name:
                arch.append(int(name.split('_')[1]))
            else:
                arch.append('skip')
        return arch

# Ecuación de Poisson
class PoissonSolver:
    def __init__(self, domain_x=(0, 1), domain_y=(0, 1)):
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.pi = np.pi
        self.factor = 1.0 / (8.0 * (self.pi**2))

    def exact_solution(self, x, y):
        return -self.factor*torch.sin(2*self.pi*x)*torch.sin(2*self.pi*y)

    def source_term(self, x, y):
        return torch.sin(2 * self.pi * x) * torch.sin(2 * self.pi * y)

    def pde_residual(self, phi, x, y):
        grad_phi = torch.autograd.grad(phi, [x, y],
                                     grad_outputs=torch.ones_like(phi),
                                     create_graph=True, retain_graph=True)
        phi_x, phi_y = grad_phi

        phi_xx = torch.autograd.grad(phi_x, x,
                                   grad_outputs=torch.ones_like(phi_x),
                                   create_graph=True)[0]
        phi_yy = torch.autograd.grad(phi_y, y,
                                   grad_outputs=torch.ones_like(phi_y),
                                   create_graph=True)[0]

        laplacian = phi_xx + phi_yy
        f = self.source_term(x, y)
        return laplacian-f

    def generate_data(self, n_interior=5000, n_boundary=2000):
        # Puntos interiores
        x_int = torch.rand(n_interior, 1)* (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        y_int = torch.rand(n_interior, 1) * (self.domain_y[1]- self.domain_y[0]) + self.domain_y[0]
        x_int.requires_grad_(True)
        y_int.requires_grad_(True)

        # Puntos de frontera
        n_per_side = n_boundary // 4

        x_left = torch.zeros(n_per_side, 1)
        y_left = torch.rand(n_per_side, 1)*(self.domain_y[1] - self.domain_y[0]) + self.domain_y[0]

        x_right = torch.ones(n_per_side, 1)
        y_right = torch.rand(n_per_side, 1)*(self.domain_y[1] - self.domain_y[0]) + self.domain_y[0]

        x_bottom = torch.rand(n_per_side, 1)*(self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        y_bottom = torch.zeros(n_per_side, 1)

        x_top = torch.rand(n_per_side, 1) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        y_top = torch.ones(n_per_side, 1)

        x_bc = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
        y_bc = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)
        phi_bc = torch.zeros_like(x_bc)  # Dirichlet homogéneo

        x_bc.requires_grad_(True)
        y_bc.requires_grad_(True)

        return {
            'interior': (x_int, y_int),
            'boundary': (x_bc, y_bc, phi_bc)
        }

    def compute_loss(self, model, data, discrete_architecture=False):
        x_int, y_int = data['interior']
        phi_pred_int = model(torch.cat([x_int, y_int], dim=1), discrete_architecture)
        residual = self.pde_residual(phi_pred_int, x_int, y_int)
        loss_pde = torch.mean(residual**2)

        x_bc, y_bc, phi_bc_true = data['boundary']
        phi_pred_bc = model(torch.cat([x_bc, y_bc], dim=1), discrete_architecture)
        loss_bc = torch.mean((phi_pred_bc - phi_bc_true)**2)

        total_loss = loss_pde + loss_bc
        return total_loss, loss_pde, loss_bc

    def compute_loss_external(self, external_model, data):
        x_int, y_int = data['interior']
        phi_pred_int = external_model(torch.cat([x_int, y_int], dim=1))
        residual = self.pde_residual(phi_pred_int, x_int, y_int)
        loss_pde = torch.mean(residual**2)

        x_bc, y_bc, phi_bc_true = data['boundary']
        phi_pred_bc = external_model(torch.cat([x_bc, y_bc], dim=1))
        loss_bc = torch.mean((phi_pred_bc - phi_bc_true)**2)

        total_loss = loss_pde + loss_bc
        return total_loss, loss_pde, loss_bc

# Heredamos pesos
def build_compact_and_transfer(supernet, input_dim=None, output_dim=None, activation='tanh', verbose=False):
    W = supernet.max_width
    input_dim = supernet.input_dim if input_dim is None else input_dim
    output_dim = supernet.output_dim if output_dim is None else output_dim
 
    arch = supernet.get_discrete_architecture()
    chosen = [(i, int(a)) for i, a in enumerate(arch) if a != 'skip']
 
    modules = [nn.Linear(input_dim, W), _activation_module(activation)]
    prev = W
    for (_, w) in chosen:
        modules += [nn.Linear(prev, w), _activation_module(activation)]
        prev = w
    modules.append(nn.Linear(prev, output_dim))
    compact = nn.Sequential(*modules)
 
    with torch.no_grad():
        lins = [m for m in compact if isinstance(m, nn.Linear)]
        lins[0].weight.copy_(supernet.input_layer.weight)
        lins[0].bias.copy_(supernet.input_layer.bias)
        prev = W
        for k, (mixed_idx, w) in enumerate(chosen):
            op = next(m for m in supernet.mixed_layers[mixed_idx].ops
                      if isinstance(m, nn.Linear) and m.out_features == w)
            lins[k + 1].weight.copy_(op.weight[:, :prev])
            lins[k + 1].bias.copy_(op.bias)
            prev = w
        lins[-1].weight.copy_(supernet.output_layer.weight[:, :prev])
        lins[-1].bias.copy_(supernet.output_layer.bias)
 
    if verbose:
        print("Compact (in->out):", [(m.in_features, m.out_features) for m in lins])
    return compact
 
 
SUPERNET_MAX_WIDTH_POISSON = 80   # max(neuron_options) de Poisson
 
def build_network_from_architecture(architecture, input_dim=2, output_dim=1, activation='tanh'):
    W = SUPERNET_MAX_WIDTH_POISSON
    widths = [w for w in architecture if w != 'skip']
    modules = [nn.Linear(input_dim, W), _activation_module(activation)]
    prev = W
    for w in widths:
        layer = nn.Linear(prev, w)
        nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        modules += [layer, _activation_module(activation)]
        prev = w
    out = nn.Linear(prev, output_dim)
    nn.init.xavier_uniform_(out.weight); nn.init.zeros_(out.bias)
    modules.append(out)
    return nn.Sequential(*modules)

# Entrenamos la inicializada aleatoriamente
def train_from_architecture(architecture, solver, n_interior=5000, n_boundary=2000,
                            iterations=3000, lr_weights=1e-3, resample_every=100,
                            disable_file_io=True, silent=True, clip_norm=5.0, device=None,
                            activation='tanh'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = build_network_from_architecture(architecture, activation=activation)
    net = net.to(device)

    def resample_interior(n, dev):
        x_int = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
        y_int = torch.rand(n, 1) * (solver.domain_y[1] - solver.domain_y[0]) + solver.domain_y[0]
        x_int.requires_grad_(True)
        y_int.requires_grad_(True)
        return x_int.to(dev), y_int.to(dev)

    x_bc, y_bc, phi_bc = solver.generate_data(n_interior=1, n_boundary=n_boundary)['boundary']  # only boundary needed
    x_bc = x_bc.to(device)
    y_bc = y_bc.to(device)
    phi_bc = phi_bc.to(device)

    x_int_dev, y_int_dev = resample_interior(n_interior, device)
    train_data = {
        'interior': (x_int_dev, y_int_dev),
        'boundary': (x_bc, y_bc, phi_bc)
    }

    optimizer = torch.optim.Adam(net.parameters(), lr=lr_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=400, factor=0.5)

    losses = []
    best_loss = float('inf')

    start_time = time.time()
    for iteration in range(iterations):
        if (iteration % resample_every) == 0:
            x_int_dev, y_int_dev = resample_interior(n_interior, device)
            train_data['interior'] = (x_int_dev, y_int_dev)

        optimizer.zero_grad()
        total_loss, lpde, lbc = solver.compute_loss_external(net, train_data)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
        optimizer.step()
        scheduler.step(total_loss.item())

        losses.append(total_loss.item())
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            if not disable_file_io:
                torch.save(net.state_dict(), 'best_network_from_arch.pth')

        if not silent and (iteration + 1) % 500 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Iteration {iteration+1}/{iterations} | Loss={total_loss.item():.6e} | LR={lr:.2e}")

    training_time = time.time() - start_time
    return net, losses, best_loss, training_time


# Nueva training function
def train_final_model(
    net,
    solver,
    n_interior=5000,
    n_boundary=2000,
    total_iterations=5000,
    lr_main=1e-4,
    resample_every=100,
    disable_file_io=True,
    silent=True,
    clip_norm=5.0,
    device=None,
    warmup_iterations=0,
    warmup_lr=None,
    mse_threshold=None,
    eval_every=50,
    activation='tanh'
):
    """
    Train a final discrete PINN under one of three regimes:
      - Scratch: random net, warmup_iterations=0
      - Inheritance: inherited net, warmup_iterations=0
      - Inheritance + warm-up: inherited net, warmup_iterations>0

    Parameters
    ----------
    net : nn.Module
        Already constructed final discrete model. It may be random or inherited.
    solver : PoissonSolver
        PDE solver / loss computer.
    total_iterations : int
        Total optimization budget, including warm-up if used.
    lr_main : float
        Main learning rate after warm-up.
    warmup_iterations : int
        Number of warm-up iterations. Use 0 for no warm-up.
    warmup_lr : float or None
        Warm-up learning rate. If None and warmup_iterations > 0, defaults to lr_main / 10.
    mse_threshold : float or None
        If provided, compute iterations needed to reach this threshold.
    eval_every : int
        Frequency for recording MSE during training.

    Returns
    -------
    net : trained model
    history : dict
        Contains losses, mses, lrs, phases, threshold info, and summary metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)

    if warmup_iterations > 0 and warmup_lr is None:
        warmup_lr = lr_main / 10.0

    main_iterations = total_iterations - warmup_iterations
    if main_iterations < 0:
        raise ValueError("warmup_iterations cannot be larger than total_iterations")

    def resample_interior(n, dev):
        x_int = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
        y_int = torch.rand(n, 1) * (solver.domain_y[1] - solver.domain_y[0]) + solver.domain_y[0]
        x_int.requires_grad_(True)
        y_int.requires_grad_(True)
        return x_int.to(dev), y_int.to(dev)

    # Fixed boundary data
    x_bc, y_bc, phi_bc = solver.generate_data(n_interior=1, n_boundary=n_boundary)['boundary']
    x_bc = x_bc.to(device)
    y_bc = y_bc.to(device)
    phi_bc = phi_bc.to(device)

    def build_train_data():
        x_int_dev, y_int_dev = resample_interior(n_interior, device)
        return {
            'interior': (x_int_dev, y_int_dev),
            'boundary': (x_bc, y_bc, phi_bc)
        }

    train_data = build_train_data()

    # Initial loss before any final optimization
    net.eval()
    with torch.enable_grad():
        initial_loss, _, _ = solver.compute_loss_external(net, train_data)

    history = {
        "losses": [],
        "mses": [],
        "mse_iters": [],
        "lrs": [],
        "phases": [],
        "initial_loss": float(initial_loss.item()),
        "final_loss": None,
        "final_mse": None,
        "best_loss": float('inf'),
        "iterations_to_threshold": None,
        "threshold_reached": False
    }

    optimizer = torch.optim.Adam(net.parameters(), lr=(warmup_lr if warmup_iterations > 0 else lr_main))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=400, factor=0.5)

    def maybe_eval_mse(iteration):
        mse_val = compute_mse_on_grid(net, solver, device=device, grid_n=100)
        history["mses"].append(float(mse_val))
        history["mse_iters"].append(iteration)

        if mse_threshold is not None and history["iterations_to_threshold"] is None:
            if mse_val <= mse_threshold:
                history["iterations_to_threshold"] = iteration
                history["threshold_reached"] = True

    # Evaluate MSE at iteration 0 if threshold metric is used
    if mse_threshold is not None:
        maybe_eval_mse(0)

    start_time = time.time()

    for iteration in range(total_iterations):
        if (iteration % resample_every) == 0:
            train_data = build_train_data()

        # Switch from warm-up to main training
        if warmup_iterations > 0 and iteration == warmup_iterations:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_main

        phase = "warmup" if iteration < warmup_iterations else "main"

        net.train()
        optimizer.zero_grad()
        total_loss, lpde, lbc = solver.compute_loss_external(net, train_data)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
        optimizer.step()

        # Apply scheduler only in main phase
        if iteration >= warmup_iterations:
            scheduler.step(total_loss.item())

        loss_value = float(total_loss.item())
        history["losses"].append(loss_value)
        history["lrs"].append(float(optimizer.param_groups[0]["lr"]))
        history["phases"].append(phase)

        if loss_value < history["best_loss"]:
            history["best_loss"] = loss_value
            if not disable_file_io:
                torch.save(net.state_dict(), 'best_final_model.pth')

        if ((iteration + 1) % eval_every == 0) or (iteration == total_iterations - 1):
            maybe_eval_mse(iteration + 1)

        if not silent and (iteration + 1) % 500 == 0:
            print(
                f"Iteration {iteration+1}/{total_iterations} | "
                f"Phase={phase} | Loss={loss_value:.6e} | "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    training_time = time.time() - start_time

    history["final_loss"] = history["losses"][-1]
    history["final_mse"] = history["mses"][-1] if len(history["mses"]) > 0 else float(
        compute_mse_on_grid(net, solver, device=device, grid_n=100)
    )
    history["training_time"] = training_time

    return net, history


def run_poisson_seed_experiment(
    seed,
    device=None,
    activation='tanh',
    resample_every_search=1,
    resample_every_final=100,
    n_interior=5000,
    n_boundary=2000,
    search_iterations=1000,
    total_iterations=5000,
    stable_iters=300,
    lr_weights_search=1e-3,
    lr_weights_train=1e-4,
    lr_alpha=1e-4,
    warmup_iterations=100,
    warmup_lr=5e-5,
    mse_threshold=None,
    eval_every=50,
    silent=True
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)

    # 1) Search stage only
    search_results = train_pinn_darts(
        resample_every=resample_every_search,
        n_interior=n_interior,
        n_boundary=n_boundary,
        search_iterations=search_iterations,
        train_iterations=total_iterations,
        switch_criterion='stable',
        stable_iters=stable_iters,
        lr_weights_search=lr_weights_search,
        lr_weights_train=lr_weights_train,
        lr_alpha=lr_alpha,
        silent=silent,
        disable_file_io=True,
        clip_norm=5.0,
        device=device,
        activation=activation,
        return_search_only=True
    )

    searched_supernet = search_results['searched_supernet']
    summary_pi = softmax_concentration(searched_supernet)   # diagnóstico: concentración del softmax
    architecture = search_results['arch']
    solver = search_results['solver']

    # 2) Final comparison
    results = run_poisson_final_comparison(
        searched_supernet=searched_supernet,
        architecture=architecture,
        solver=solver,
        device=device,
        activation=activation,
        total_iterations=total_iterations,
        lr_main=lr_weights_train,
        warmup_iterations=warmup_iterations,
        warmup_lr=warmup_lr,
        n_interior=n_interior,
        n_boundary=n_boundary,
        resample_every=resample_every_final,
        mse_threshold=mse_threshold,
        eval_every=eval_every,
        silent=silent,
        verbose_inheritance=False
    )

    # 3) Compact summary
    summary = {
        "seed": seed,
        "architecture": architecture,
        "pi_max": summary_pi,
        "search_time": search_results["architecture_search_time"],
        "scratch": results["scratch"]["history"],
        "inheritance": results["inheritance"]["history"],
        "inheritance_warmup": results["inheritance_warmup"]["history"]
    }

    return summary, results


def summarize_poisson_seed_summaries(seed_summaries):
    for run in seed_summaries:
        print(f"\n==============================")
        print(f"SEED {run['seed']} | ARCH = {run['architecture']}")
        print(f"Search time: {run['search_time']:.2f} s")

        for method in ["scratch", "inheritance", "inheritance_warmup"]:
            h = run[method]
            print(f"\n  {method.upper()}")
            print(f"    Initial loss: {h['initial_loss']:.6e}")
            print(f"    Final loss:   {h['final_loss']:.6e}")
            print(f"    Final MSE:    {h['final_mse']:.6e}")
            print(f"    Best loss:    {h['best_loss']:.6e}")
            print(f"    Iter->thr:    {h['iterations_to_threshold']}")


def aggregate_poisson_metrics(seed_summaries):
    methods = ["scratch", "inheritance", "inheritance_warmup"]
    metrics = ["initial_loss", "final_loss", "final_mse"]

    for method in methods:
        print(f"\n===== {method.upper()} =====")
        for metric in metrics:
            vals = [run[method][metric] for run in seed_summaries]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            print(f"{metric}: {mean_val:.6e} ± {std_val:.6e}")

        thr_vals = [
            run[method]["iterations_to_threshold"]
            for run in seed_summaries
            if run[method]["iterations_to_threshold"] is not None
        ]
        reached = sum(run[method]["threshold_reached"] for run in seed_summaries)
        total = len(seed_summaries)
        print(f"threshold reached: {reached}/{total}")

        if len(thr_vals) > 0:
            print(f"iterations_to_threshold: {np.mean(thr_vals):.2f} ± {np.std(thr_vals):.2f}")
        else:
            print("iterations_to_threshold: none")


def run_poisson_final_comparison(
    searched_supernet,
    architecture,
    solver,
    device=None,
    activation='tanh',
    total_iterations=5000,
    lr_main=1e-4,
    warmup_iterations=300,
    warmup_lr=1e-5,
    n_interior=5000,
    n_boundary=2000,
    resample_every=100,
    mse_threshold=None,
    eval_every=50,
    silent=True,
    verbose_inheritance=False
):
    """
    Run the 3-way final training comparison for Poisson:
      1) Scratch
      2) Inheritance
      3) Inheritance + Warm-up

    Parameters
    ----------
    searched_supernet : nn.Module
        The trained continuous supernetwork returned by the search stage.
    architecture : list
        Final discrete architecture extracted after search.
    solver : PoissonSolver
        Solver used for PDE loss and MSE evaluation.
    device : torch.device or None
    activation : str
    total_iterations : int
        Total final training budget for every method.
    lr_main : float
        Main learning rate.
    warmup_iterations : int
        Warm-up iterations for the warm-up variant only.
    warmup_lr : float
        Reduced learning rate for warm-up.
    mse_threshold : float or None
        Optional threshold for iterations-to-threshold metric.

    Returns
    -------
    results : dict
        Contains trained models and histories for the 3 methods.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    # -------------------------
    # 1) Scratch
    # -------------------------
    scratch_model = build_scratch_model(
        architecture=architecture,
        device=device,
        activation=activation
    )

    scratch_model, scratch_history = train_final_model(
        net=scratch_model,
        solver=solver,
        n_interior=n_interior,
        n_boundary=n_boundary,
        total_iterations=total_iterations,
        lr_main=lr_main,
        resample_every=resample_every,
        silent=silent,
        device=device,
        warmup_iterations=0,
        warmup_lr=None,
        mse_threshold=mse_threshold,
        eval_every=eval_every,
        activation=activation
    )

    results["scratch"] = {
        "model": scratch_model,
        "history": scratch_history
    }

    # -------------------------
    # 2) Inheritance
    # -------------------------
    inherited_model = build_inherited_model(
        supernet=searched_supernet,
        device=device,
        activation=activation,
        verbose=verbose_inheritance
    )

    inherited_model, inherited_history = train_final_model(
        net=inherited_model,
        solver=solver,
        n_interior=n_interior,
        n_boundary=n_boundary,
        total_iterations=total_iterations,
        lr_main=lr_main,
        resample_every=resample_every,
        silent=silent,
        device=device,
        warmup_iterations=0,
        warmup_lr=None,
        mse_threshold=mse_threshold,
        eval_every=eval_every,
        activation=activation
    )

    results["inheritance"] = {
        "model": inherited_model,
        "history": inherited_history
    }

    # -------------------------
    # 3) Inheritance + Warm-up
    # -------------------------
    inherited_warm_model = build_inherited_model(
        supernet=searched_supernet,
        device=device,
        activation=activation,
        verbose=verbose_inheritance
    )

    inherited_warm_model, inherited_warm_history = train_final_model(
        net=inherited_warm_model,
        solver=solver,
        n_interior=n_interior,
        n_boundary=n_boundary,
        total_iterations=total_iterations,
        lr_main=lr_main,
        resample_every=resample_every,
        silent=silent,
        device=device,
        warmup_iterations=warmup_iterations,
        warmup_lr=warmup_lr,
        mse_threshold=mse_threshold,
        eval_every=eval_every,
        activation=activation
    )

    results["inheritance_warmup"] = {
        "model": inherited_warm_model,
        "history": inherited_warm_history
    }

    return results


#MSE
def compute_mse_on_grid(model, solver, device=None, grid_n=100):
    if device is None:
        device = next(model.parameters()).device
    x = torch.linspace(0, 1, grid_n)
    y = torch.linspace(0, 1, grid_n)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)
    with torch.no_grad():
        model.eval()
        if isinstance(model, PINNDARTSNet):
            pred = model(torch.cat([X_flat, Y_flat], dim=1), discrete_architecture=True)
        else:
            pred = model(torch.cat([X_flat, Y_flat], dim=1))
        exact = solver.exact_solution(X_flat, Y_flat)
        mse = torch.mean((pred - exact)**2).item()
    return mse

# Entrenar superred
def train_pinn_darts(
    resample_every=1, n_interior=5000, n_boundary=2000,
    search_iterations=1000, train_iterations=5000,
    switch_criterion='stable', stable_iters=300,
    lr_weights_search=1e-3, lr_weights_train=1e-4, lr_alpha=1e-4,
    silent=True, disable_file_io=True, clip_norm=5.0, device=None,
    activation='tanh',
    return_search_only=False):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_layers = 7
    neuron_options = [50, 60, 70, 80]

    model = PINNDARTSNet(max_layers=max_layers, neuron_options=neuron_options, activation=activation).to(device)
    solver = PoissonSolver()

    base_boundary = solver.generate_data(n_interior=1, n_boundary=n_boundary)['boundary']
    x_bc, y_bc, phi_bc = base_boundary
    x_bc = x_bc.to(device)
    y_bc = y_bc.to(device)
    phi_bc = phi_bc.to(device)

    # Datos fijos para la actualización de alphas
    arch_data_raw = solver.generate_data(n_interior=n_interior, n_boundary=n_boundary)
    x_arch_int, y_arch_int = arch_data_raw['interior']
    x_arch_int = x_arch_int.to(device)
    y_arch_int = y_arch_int.to(device)
    x_arch_int.requires_grad_(True)
    y_arch_int.requires_grad_(True)
    arch_data = {
        'interior': (x_arch_int, y_arch_int),
        'boundary': (x_bc, y_bc, phi_bc)
    }

    optimizer_weights = torch.optim.Adam([p for n, p in model.named_parameters() if 'alpha' not in n],
                                         lr=lr_weights_search)
    optimizer_alpha = torch.optim.Adam(model.alpha_parameters, lr=lr_alpha)
    scheduler_weights = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_weights, patience=400, factor=0.5)

    # Arrays de pérdidas
    losses_search_weights = []
    losses_search_alphas = []
    best_loss = float('inf')
    best_architecture = None

    stable_count = 0
    last_arch = model.get_discrete_architecture()

    start_time_search = time.time()

    for iteration in range(search_iterations):
        def resample_interior(n, dev):
            x_int = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
            y_int = torch.rand(n, 1) * (solver.domain_y[1] - solver.domain_y[0]) + solver.domain_y[0]
            x_int.requires_grad_(True)
            y_int.requires_grad_(True)
            return x_int.to(dev), y_int.to(dev)

        if (iteration % resample_every) == 0:
            x_train_int, y_train_int = resample_interior(n_interior, device)
        train_data = {
            'interior': (x_train_int, y_train_int),
            'boundary': (x_bc, y_bc, phi_bc)
        }

        # Actualizamos pesos
        model.train()
        optimizer_weights.zero_grad()
        optimizer_alpha.zero_grad()

        total_loss_w, loss_pde_w, loss_bc_w = solver.compute_loss(model, train_data, discrete_architecture=False)
        total_loss_w.backward()


        for p in model.alpha_parameters:
            p.grad = None

        torch.nn.utils.clip_grad_norm_([p for n,p in model.named_parameters() if 'alpha' not in n], clip_norm)
        optimizer_weights.step()
        scheduler_weights.step(total_loss_w.item())

        losses_search_weights.append(total_loss_w.item())

        # Actualizamos los pesos arquitectónicos
        optimizer_alpha.zero_grad()
        total_loss_a, loss_pde_a, loss_bc_a = solver.compute_loss(model, arch_data, discrete_architecture=False)
        total_loss_a.backward()


        for param in model.parameters():
            if not any(param is a for a in model.alpha_parameters):
                param.grad = None

        optimizer_alpha.step()
        losses_search_alphas.append(total_loss_a.item())

        current_arch = model.get_discrete_architecture()
        if current_arch == last_arch:
            stable_count += 1
        else:
            stable_count = 0
            last_arch = current_arch

        if total_loss_w.item() < best_loss:
            best_loss = total_loss_w.item()
            best_architecture = model.get_discrete_architecture()

        if not silent and (iteration + 1) % 200 == 0:
            lr_w = optimizer_weights.param_groups[0]["lr"]
            lr_a = optimizer_alpha.param_groups[0]["lr"]
            print(f"[Search] Iter {iteration+1}/{search_iterations} | Loss_w={total_loss_w.item():.3e} | Loss_a={total_loss_a.item():.3e} | LR_w={lr_w:.2e} | LR_a={lr_a:.2e}")

        if stable_count >= stable_iters:
            if not silent:
                print(f"Arquitectura estable detectada tras {iteration+1} iteraciones: {current_arch}")
            break

    architecture_search_time = time.time() - start_time_search
    arch = model.get_discrete_architecture()

    if return_search_only:
        return {
            'searched_supernet': model,
            'solver': solver,
            'losses_search_weights': losses_search_weights,
            'losses_search_alphas': losses_search_alphas,
            'best_architecture': best_architecture,
            'best_loss_search': best_loss,
            'architecture_search_time': architecture_search_time,
            'arch': arch
        }

    if not silent:
        print(f"Switching to compact architecture: {arch}")

    compact = build_compact_and_transfer(model, input_dim=model.input_dim, output_dim=model.output_dim, activation=activation, verbose=(not silent))
    compact = compact.to(device)

    optimizer_compact = torch.optim.Adam(compact.parameters(), lr=lr_weights_train)
    scheduler_compact = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_compact, patience=400, factor=0.5)

    x_bc_c, y_bc_c, phi_bc_c = x_bc, y_bc, phi_bc

    def resample_interior_comp(n, dev):
        x_int = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
        y_int = torch.rand(n, 1) * (solver.domain_y[1] - solver.domain_y[0]) + solver.domain_y[0]
        x_int.requires_grad_(True)
        y_int.requires_grad_(True)
        return x_int.to(dev), y_int.to(dev)

    losses_compact = []
    start_time_compact = time.time()
    for iteration in range(train_iterations):
        if (iteration % resample_every) == 0:
            x_int_c, y_int_c = resample_interior_comp(n_interior, device)
        train_data_c = {
            'interior': (x_int_c, y_int_c),
            'boundary': (x_bc_c, y_bc_c, phi_bc_c)
        }

        optimizer_compact.zero_grad()
        total_loss_c, lpde_c, lbc_c = solver.compute_loss_external(compact, train_data_c)
        total_loss_c.backward()
        torch.nn.utils.clip_grad_norm_(compact.parameters(), clip_norm)
        optimizer_compact.step()
        scheduler_compact.step(total_loss_c.item())

        losses_compact.append(total_loss_c.item())

        if not silent and (iteration + 1) % 500 == 0:
            lr_c = optimizer_compact.param_groups[0]["lr"]
            print(f"[Compact train] Iter {iteration+1}/{train_iterations} | Loss={total_loss_c.item():.3e} | LR={lr_c:.2e}")

    compact_training_time = time.time() - start_time_compact

    final_model = compact
    final_loss = losses_compact[-1] if len(losses_compact) > 0 else float('nan')

    mse_compact = compute_mse_on_grid(final_model, solver, device=device, grid_n=100)

    return {
        'final_model': final_model,
        'solver': solver,
        'losses_search_weights': losses_search_weights,
        'losses_search_alphas': losses_search_alphas,
        'losses_compact': losses_compact,
        'best_architecture': best_architecture,
        'best_loss_search': best_loss,
        'final_loss_compact': final_loss,
        'architecture_search_time': architecture_search_time,
        'compact_training_time': compact_training_time,
        'mse_compact': mse_compact,
        'arch': arch
    }


# Gráficas
def evaluate_and_plot(model_or_net, solver, best_architecture, final_loss, best_loss=None, title_suffix="", disable_file_io=True):
    device = next(model_or_net.parameters()).device

    # Crear malla para evaluación
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)

    inputs = torch.cat([X_flat, Y_flat], dim=1)
    with torch.no_grad():
        model_or_net.eval()
        if isinstance(model_or_net, PINNDARTSNet):
            prediction = model_or_net(inputs, discrete_architecture=True)
        else:
            prediction = model_or_net(inputs)
        exact = solver.exact_solution(X_flat, Y_flat)

    # Convertir a formas 2D para plotting
    pred_2d = prediction.reshape(100, 100).cpu().numpy()
    exact_2d = exact.reshape(100, 100).cpu().numpy()
    error_2d = np.abs(pred_2d - exact_2d)

    # Calcular error L2 relativo
    mse = torch.mean((prediction - exact)**2)
    l2_rel = torch.sqrt(mse) / torch.sqrt(torch.mean(exact**2))
    l2_rel_value = l2_rel.item()

    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Predicción del modelo
    im1 = axes[0,0].contourf(X.cpu().numpy(), Y.cpu().numpy(), pred_2d, levels=50, cmap='viridis')
    axes[0,0].set_title('Solución del Modelo')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])

    # Solución exacta
    im2 = axes[0,1].contourf(X.cpu().numpy(), Y.cpu().numpy(), exact_2d, levels=50, cmap='viridis')
    axes[0,1].set_title('Solución Exacta')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])

    # Error absoluto
    im3 = axes[1,0].contourf(X.cpu().numpy(), Y.cpu().numpy(), error_2d, levels=50, cmap='hot')
    axes[1,0].set_title(f'Error Absoluto (L2 rel: {l2_rel_value:.2e})')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1,0])

    axes[1,1].axis('off')
    arch_text = "Arquitectura Final:\n" + (str(best_architecture) if best_architecture is not None else "N/A")
    axes[1,1].text(0.1, 0.9, arch_text, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1,1].text(0.1, 0.6, f'Pérdida final: {final_loss:.6f}', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    if best_loss is not None:
        axes[1,1].text(0.1, 0.3, f'Mejor pérdida: {best_loss:.6f}', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    if not disable_file_io:
        plt.savefig(f'poisson_results{title_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig, l2_rel_value


def build_scratch_model(architecture, device=None, activation='tanh'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_network_from_architecture(architecture, activation=activation).to(device)
    return net


def build_inherited_model(supernet, device=None, activation='tanh', verbose=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_compact_and_transfer(supernet, activation=activation, verbose=verbose).to(device)
    return net


def summarize_final_comparison(results):
    """
    Print a compact summary of the 3-way comparison.
    """
    for method_name, payload in results.items():
        h = payload["history"]
        print(f"\n=== {method_name.upper()} ===")
        print(f"Initial loss:            {h['initial_loss']:.6e}")
        print(f"Final loss:              {h['final_loss']:.6e}")
        print(f"Final MSE:               {h['final_mse']:.6e}")
        print(f"Best loss:               {h['best_loss']:.6e}")
        print(f"Threshold reached:       {h['threshold_reached']}")
        print(f"Iterations to threshold: {h['iterations_to_threshold']}")
        print(f"Training time (s):       {h['training_time']:.2f}")
