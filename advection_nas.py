"""Benchmark de ADVECCIÓN 1D (ReLU): supernet DARTS, búsqueda, discretización,
herencia de pesos, scratch y comparación final. Importar como:
    import advection_nas as ADV
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime
import os
import glob

# Semilla
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#Diagnósitco
def softmax_concentration(supernet):
    """pi_max por capa buscable: cuanto 'pesa' la rama ganadora al acabar la
    búsqueda. 1.0 = one-hot (herencia exacta). 0.2 = uniforme con 5 ramas."""
    out = []
    with torch.no_grad():
        for i, a in enumerate(supernet.alpha_parameters):
            pi = torch.softmax(a, dim=0)
            k = int(torch.argmax(pi))
            out.append((i, supernet.op_names_per_layer[i][k], round(float(pi[k]), 3)))
    print("pi_max por capa:", out, "| min:", min(p for _, _, p in out))
    return out


# Capas mixtas
# --------------------------
class MixedOperation(nn.Module):
    def __init__(self, input_size,output_sizes, max_width):
        super().__init__()
        self.output_sizes = list(output_sizes)
        self.max_width = max(self.output_sizes)
        self.ops = nn.ModuleList()
        for out_size in self.output_sizes:
            self.ops.append(nn.Linear(input_size, out_size))
        self.ops.append(nn.Identity())

    def forward(self, x, alphas):
        out = 0.0
        for idx, op in enumerate(self.ops):
            op_out = op(x)
            if isinstance(op,nn.Linear):
                op_out = F.relu(op_out)
            if op_out.size(1) < self.max_width:
                pad = torch.zeros(op_out.size(0),self.max_width -op_out.size(1),
                                  device=op_out.device, dtype=op_out.dtype)
                op_out = torch.cat([op_out, pad], dim=1)
            out = out + alphas[idx] * op_out
        return out

#Superred
class PINNDARTSNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, max_layers=8,neuron_options=[10,20,30,40]):
        super().__init__()
        self.max_layers = max_layers
        self.neuron_options = list(neuron_options)
        self.max_width = max(self.neuron_options)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_layer = nn.Linear(input_dim, self.max_width)
        self.mixed_layers = nn.ModuleList()
        for _ in range(max_layers - 1):
            self.mixed_layers.append(MixedOperation(self.max_width, self.neuron_options, self.max_width))

        self.output_layer = nn.Linear(self.max_width, output_dim)

        self.alpha_parameters = nn.ParameterList()
        self.op_names_per_layer = []
        num_ops = len(self.neuron_options) + 1
        for _ in range(max_layers - 1):
            p = nn.Parameter(torch.randn(num_ops)* 1e-3)
            self.alpha_parameters.append(p)
            names = [f'linear_{s}' for s in self.neuron_options] + ['skip']
            self.op_names_per_layer.append(names)

    def forward(self, x, discrete_architecture=False):
        x = F.relu(self.input_layer(x))
        for i, mixed_layer in enumerate(self.mixed_layers):
            if discrete_architecture:
                raw = self.alpha_parameters[i]
                idx_max = torch.argmax(raw).item()
                op = mixed_layer.ops[idx_max]
                x = op(x)
                if isinstance(op, nn.Linear):
                    x = F.relu(x)
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

# Ecuación de advección
class AdvectionSolver:
    def __init__(self, v=1.2, domain_x=(-1, 1), domain_t=(0, 1)):
        self.v = v
        self.domain_x = domain_x
        self.domain_t = domain_t

    def front(self, x, w=0.4):
        condi1 = (x<= -1)
        condi2 = (x >=-1 + 2 * w)
        interm = 0.5 + (x + 1- w) * ((x+ 1-w)**2-3 *w**2)/(4* w**3)
        return torch.where(condi1, torch.ones_like(x),
                          torch.where(condi2, torch.zeros_like(x), interm))

    def exact_solution(self, x,t,w=0.4):
        return self.front(x - self.v * t, w)

    def pde_residual(self, u, x, t):
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        return u_t + self.v * u_x

    def generate_data(self, n_interior=1000, n_initial=200, n_boundary=100, device=None):
        x_int = torch.rand(n_interior, 1)* (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        t_int = torch.rand(n_interior, 1)* (self.domain_t[1] - self.domain_t[0])
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)

        x_ic = torch.rand(n_initial, 1) * (self.domain_x[1] - self.domain_x[0]) + self.domain_x[0]
        t_ic = torch.zeros(n_initial, 1)
        u_ic = self.front(x_ic)
        x_ic.requires_grad_(True)
        t_ic.requires_grad_(True)

        t_bc = torch.rand(n_boundary, 1) * (self.domain_t[1] - self.domain_t[0])
        x_bc = torch.ones(n_boundary, 1) * self.domain_x[0]
        u_bc = torch.ones(n_boundary, 1)
        x_bc.requires_grad_(True)
        t_bc.requires_grad_(True)

        if device is not None:
            x_int = x_int.to(device); t_int = t_int.to(device)
            x_ic = x_ic.to(device); t_ic = t_ic.to(device); u_ic = u_ic.to(device)
            x_bc = x_bc.to(device); t_bc = t_bc.to(device); u_bc = u_bc.to(device)

        return {
            'interior': (x_int, t_int),
            'initial': (x_ic, t_ic, u_ic),
            'boundary': (x_bc, t_bc, u_bc)
        }

    def compute_loss(self, model, data, discrete_architecture=False):
        x_int, t_int = data['interior']
        u_pred_int = model(torch.cat([x_int, t_int], dim=1), discrete_architecture)
        residual = self.pde_residual(u_pred_int, x_int, t_int)
        loss_pde = torch.mean(residual**2)

        x_ic, t_ic, u_ic_true = data['initial']
        u_pred_ic = model(torch.cat([x_ic, t_ic], dim=1), discrete_architecture)
        loss_ic = torch.mean((u_pred_ic - u_ic_true)**2)

        x_bc, t_bc, u_bc_true = data['boundary']
        u_pred_bc = model(torch.cat([x_bc, t_bc], dim=1), discrete_architecture)
        loss_bc = torch.mean((u_pred_bc - u_bc_true)**2)

        total_loss = loss_pde + loss_ic + loss_bc
        return total_loss, loss_pde, loss_ic, loss_bc

    def compute_loss_external(self, external_model, data):
        x_int, t_int = data['interior']
        u_pred_int = external_model(torch.cat([x_int, t_int], dim=1))
        residual = self.pde_residual(u_pred_int, x_int, t_int)
        loss_pde = torch.mean(residual**2)

        x_ic, t_ic, u_ic_true = data['initial']
        u_pred_ic = external_model(torch.cat([x_ic, t_ic], dim=1))
        loss_ic = torch.mean((u_pred_ic - u_ic_true)**2)

        x_bc, t_bc, u_bc_true = data['boundary']
        u_pred_bc = external_model(torch.cat([x_bc, t_bc], dim=1))
        loss_bc = torch.mean((u_pred_bc - u_bc_true)**2)

        total_loss = loss_pde + loss_ic + loss_bc
        return total_loss, loss_pde, loss_ic, loss_bc

# Heredamos pesos
def build_compact_and_transfer(supernet, input_dim=None, output_dim=None, verbose=False):
    """Red discreta con los pesos heredados de la supernet.
    Estructura: Linear(in, W) -> ReLU -> [Linear(prev, w_k) -> ReLU]_k -> Linear(w_n, out)
    Cada capa elegida recibe EXACTAMENTE los pesos de su rama en la supernet,
    recortando las columnas que en la supernet multiplicaban al zero-padding.
    Si el softmax fuese one-hot, esta red computa la misma funcion que la supernet.
    """
    W = supernet.max_width
    input_dim = supernet.input_dim if input_dim is None else input_dim
    output_dim = supernet.output_dim if output_dim is None else output_dim
 
    arch = supernet.get_discrete_architecture()
    chosen = [(i, int(a)) for i, a in enumerate(arch) if a != 'skip']
 
    modules = [nn.Linear(input_dim, W), nn.ReLU()]
    prev = W
    for (_, w) in chosen:
        modules += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    modules.append(nn.Linear(prev, output_dim))
    compact = nn.Sequential(*modules)
 
    with torch.no_grad():
        lins = [m for m in compact if isinstance(m, nn.Linear)]
        # capa de entrada: copia completa
        lins[0].weight.copy_(supernet.input_layer.weight)
        lins[0].bias.copy_(supernet.input_layer.bias)
        # capas elegidas: rama seleccionada, columnas [:prev]
        prev = W
        for k, (mixed_idx, w) in enumerate(chosen):
            op = next(m for m in supernet.mixed_layers[mixed_idx].ops
                      if isinstance(m, nn.Linear) and m.out_features == w)
            lins[k + 1].weight.copy_(op.weight[:, :prev])
            lins[k + 1].bias.copy_(op.bias)
            prev = w
        # capa de salida: columnas [:prev]
        lins[-1].weight.copy_(supernet.output_layer.weight[:, :prev])
        lins[-1].bias.copy_(supernet.output_layer.bias)
 
    if verbose:
        print("Compact (in->out):", [(m.in_features, m.out_features) for m in lins])
    return compact
 
 
SUPERNET_MAX_WIDTH_ADVECTION = 100   # max(neuron_options) de advección
 
def build_network_from_architecture(architecture, input_dim=2, output_dim=1):
    """Misma forma que la red heredada, pero inicializada al azar (scratch)."""
    W = SUPERNET_MAX_WIDTH_ADVECTION
    widths = [w for w in architecture if w != 'skip']
    modules = [nn.Linear(input_dim, W), nn.ReLU()]
    prev = W
    for w in widths:
        modules += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    modules.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*modules)


# Entrenamos arquitectura con inicialización aleatoria
def train_from_architecture(architecture, solver, n_interior=1000, n_initial=200, n_boundary=100,
                            iterations=5000, lr_weights=1e-4, resample_every=100,
                            disable_file_io=True, silent=True, clip_norm=1.0, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net= build_network_from_architecture(architecture)
    net=net.to(device)

    base = solver.generate_data(n_interior=n_interior, n_initial=n_initial, n_boundary=n_boundary, device=device)
    x_ic, t_ic, u_ic = base['initial']
    x_bc, t_bc, u_bc = base['boundary']

    def resample_interior(n, dev):
        x_int = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
        t_int = torch.rand(n, 1) * (solver.domain_t[1] - solver.domain_t[0])
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)
        return x_int.to(dev), t_int.to(dev)

    x_int_dev, t_int_dev = resample_interior(n_interior, device)
    train_data = {
        'interior': (x_int_dev, t_int_dev),
        'initial': (x_ic.to(device), t_ic.to(device), u_ic.to(device)),
        'boundary': (x_bc.to(device), t_bc.to(device), u_bc.to(device))
    }

    optimizer = torch.optim.Adam(net.parameters(), lr=lr_weights)
    losses = []
    best_loss = float('inf')

    start_time = time.time()
    for iteration in range(iterations):
        if (iteration % resample_every) == 0 and iteration > 0:
            x_int_dev, t_int_dev = resample_interior(n_interior, device)
            train_data['interior'] = (x_int_dev, t_int_dev)

        optimizer.zero_grad()
        total_loss, lpde, lic, lbc = solver.compute_loss_external(net, train_data)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
        optimizer.step()

        losses.append(total_loss.item())
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            if not disable_file_io:
                torch.save(net.state_dict(), 'best_network_from_arch.pth')

    training_time = time.time() - start_time
    return net, losses, best_loss, training_time


def train_final_model_advection(
    net,
    solver,
    x_eval_all,
    t_eval_all,
    n_interior=2000,
    n_initial=200,
    n_boundary=100,
    total_iterations=5000,
    lr_main=1e-4,
    resample_every=100,
    disable_file_io=True,
    silent=True,
    clip_norm=1.0,
    device=None,
    warmup_iterations=0,
    warmup_lr=None,
    mse_threshold=None,
    eval_every=50
):
    """
    Train a final discrete Advection PINN under one of three regimes:
      - Scratch
      - Inheritance
      - Inheritance + Warm-up
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)

    if warmup_iterations > 0 and warmup_lr is None:
        warmup_lr = lr_main / 10.0

    main_iterations = total_iterations - warmup_iterations
    if main_iterations < 0:
        raise ValueError("warmup_iterations cannot be larger than total_iterations")

    base = solver.generate_data(
        n_interior=n_interior,
        n_initial=n_initial,
        n_boundary=n_boundary,
        device=device
    )

    x_ic, t_ic, u_ic = base['initial']
    x_bc, t_bc, u_bc = base['boundary']

    def resample_interior(n, dev):
        x_int = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
        t_int = torch.rand(n, 1) * (solver.domain_t[1] - solver.domain_t[0])
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)
        return x_int.to(dev), t_int.to(dev)

    x_int_dev, t_int_dev = resample_interior(n_interior, device)
    train_data = {
        'interior': (x_int_dev, t_int_dev),
        'initial': (x_ic.to(device), t_ic.to(device), u_ic.to(device)),
        'boundary': (x_bc.to(device), t_bc.to(device), u_bc.to(device))
    }

    net.eval()
    with torch.enable_grad():
        initial_loss, _, _, _ = solver.compute_loss_external(net, train_data)

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

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=(warmup_lr if warmup_iterations > 0 else lr_main)
    )

    def maybe_eval_mse(iteration):
        mse_val = compute_test_mse(
            net, solver,
            x_eval_all, t_eval_all,
            device=device,
            discrete_architecture=False
        )
        history["mses"].append(float(mse_val))
        history["mse_iters"].append(iteration)

        if mse_threshold is not None and history["iterations_to_threshold"] is None:
            if mse_val <= mse_threshold:
                history["iterations_to_threshold"] = iteration
                history["threshold_reached"] = True

    if mse_threshold is not None:
        maybe_eval_mse(0)

    start_time = time.time()

    for iteration in range(total_iterations):
        if (iteration % resample_every) == 0 and iteration > 0:
            x_int_dev, t_int_dev = resample_interior(n_interior, device)
            train_data['interior'] = (x_int_dev, t_int_dev)

        if warmup_iterations > 0 and iteration == warmup_iterations:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_main

        phase = "warmup" if iteration < warmup_iterations else "main"

        net.train()
        optimizer.zero_grad()
        total_loss, lpde, lic, lbc = solver.compute_loss_external(net, train_data)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
        optimizer.step()

        loss_value = float(total_loss.item())
        history["losses"].append(loss_value)
        history["lrs"].append(float(optimizer.param_groups[0]["lr"]))
        history["phases"].append(phase)

        if loss_value < history["best_loss"]:
            history["best_loss"] = loss_value
            if not disable_file_io:
                torch.save(net.state_dict(), 'best_final_model_advection.pth')

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
    history["final_mse"] = history["mses"][-1] if len(history["mses"]) > 0 else compute_test_mse(
        net, solver, x_eval_all, t_eval_all, device=device, discrete_architecture=False
    )
    history["training_time"] = training_time

    return net, history


def build_scratch_model_advection(architecture, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_network_from_architecture(architecture).to(device)
    return net


def build_inherited_model_advection(supernet, device=None, verbose=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_compact_and_transfer(
        supernet,
        input_dim=supernet.input_dim,
        output_dim=supernet.output_dim,
        verbose=verbose
    ).to(device)
    return net


def run_advection_final_comparison(
    searched_supernet,
    architecture,
    solver,
    x_eval_all,
    t_eval_all,
    device=None,
    total_iterations=5000,
    lr_main=1e-4,
    warmup_iterations=100,
    warmup_lr=5e-5,
    n_interior=2000,
    n_initial=200,
    n_boundary=100,
    resample_every=100,
    mse_threshold=None,
    eval_every=50,
    silent=True,
    verbose_inheritance=False
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    # Scratch
    scratch_model = build_scratch_model_advection(
        architecture=architecture,
        device=device
    )
    scratch_model, scratch_history = train_final_model_advection(
        net=scratch_model,
        solver=solver,
        x_eval_all=x_eval_all,
        t_eval_all=t_eval_all,
        n_interior=n_interior,
        n_initial=n_initial,
        n_boundary=n_boundary,
        total_iterations=total_iterations,
        lr_main=lr_main,
        resample_every=resample_every,
        silent=silent,
        device=device,
        warmup_iterations=0,
        warmup_lr=None,
        mse_threshold=mse_threshold,
        eval_every=eval_every
    )
    results["scratch"] = {"model": scratch_model, "history": scratch_history}

    # Inheritance
    inherited_model = build_inherited_model_advection(
        supernet=searched_supernet,
        device=device,
        verbose=verbose_inheritance
    )
    inherited_model, inherited_history = train_final_model_advection(
        net=inherited_model,
        solver=solver,
        x_eval_all=x_eval_all,
        t_eval_all=t_eval_all,
        n_interior=n_interior,
        n_initial=n_initial,
        n_boundary=n_boundary,
        total_iterations=total_iterations,
        lr_main=lr_main,
        resample_every=resample_every,
        silent=silent,
        device=device,
        warmup_iterations=0,
        warmup_lr=None,
        mse_threshold=mse_threshold,
        eval_every=eval_every
    )
    results["inheritance"] = {"model": inherited_model, "history": inherited_history}

    # Inheritance + Warm-up
    inherited_warm_model = build_inherited_model_advection(
        supernet=searched_supernet,
        device=device,
        verbose=verbose_inheritance
    )
    inherited_warm_model, inherited_warm_history = train_final_model_advection(
        net=inherited_warm_model,
        solver=solver,
        x_eval_all=x_eval_all,
        t_eval_all=t_eval_all,
        n_interior=n_interior,
        n_initial=n_initial,
        n_boundary=n_boundary,
        total_iterations=total_iterations,
        lr_main=lr_main,
        resample_every=resample_every,
        silent=silent,
        device=device,
        warmup_iterations=warmup_iterations,
        warmup_lr=warmup_lr,
        mse_threshold=mse_threshold,
        eval_every=eval_every
    )
    results["inheritance_warmup"] = {
        "model": inherited_warm_model,
        "history": inherited_warm_history
    }

    return results


def run_advection_seed_experiment(
    seed,
    x_eval_all,
    t_eval_all,
    device=None,
    resample_every_search=100,
    resample_every_final=100,
    n_interior=2000,
    n_initial=200,
    n_boundary=100,
    search_max_iterations=1000,
    total_iterations=5000,
    stable_iters=300,
    lr_weights_search=1e-3,
    lr_weights_train=1e-4,
    lr_alpha=1e-4,
    max_layers=6,
    neuron_options=None,
    warmup_iterations=100,
    warmup_lr=5e-5,
    mse_threshold=None,
    eval_every=50,
    silent=True
):
    if neuron_options is None:
        neuron_options = [25, 50, 75, 100]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)

    search_results = train_pinn_darts(
        resample_every=resample_every_search,
        n_interior=n_interior,
        n_initial=n_initial,
        n_boundary=n_boundary,
        switch_criterion='stable',
        discrete_at=1000,
        stable_iters=stable_iters,
        search_max_iterations=search_max_iterations,
        post_switch_iterations=total_iterations,
        lr_weights_search=lr_weights_search,
        lr_weights_compact=lr_weights_train,
        lr_alpha=lr_alpha,
        max_layers=max_layers,
        neuron_options=neuron_options,
        silent=silent,
        disable_file_io=True,
        clip_norm=1.0,
        device=device,
        return_search_only=True
    )

    searched_supernet = search_results['searched_supernet']
    summary_pi = softmax_concentration(searched_supernet)
    architecture = search_results['arch']
    solver = search_results['solver']

    results = run_advection_final_comparison(
        searched_supernet=searched_supernet,
        architecture=architecture,
        solver=solver,
        x_eval_all=x_eval_all,
        t_eval_all=t_eval_all,
        device=device,
        total_iterations=total_iterations,
        lr_main=lr_weights_train,
        warmup_iterations=warmup_iterations,
        warmup_lr=warmup_lr,
        n_interior=n_interior,
        n_initial=n_initial,
        n_boundary=n_boundary,
        resample_every=resample_every_final,
        mse_threshold=mse_threshold,
        eval_every=eval_every,
        silent=silent,
        verbose_inheritance=False
    )

    summary = {
        "seed": seed,
        "architecture": architecture,
        "pi_max": summary_pi,
        "stable_stop": search_results.get("stable_stop"),
        "search_iterations_done": search_results.get("search_iterations_done"),
        "search_time": search_results["architecture_search_time"],
        "scratch": results["scratch"]["history"],
        "inheritance": results["inheritance"]["history"],
        "inheritance_warmup": results["inheritance_warmup"]["history"]
    }

    return summary, results


def summarize_advection_seed_summaries(seed_summaries):
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


# MSE
def compute_test_mse(model, solver, x_points, t_points, device, discrete_architecture=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.cat([x_points.to(device), t_points.to(device)], dim=1)
        preds = model(inputs) if not isinstance(model, PINNDARTSNet) else model(inputs, discrete_architecture=discrete_architecture)
        exact = solver.exact_solution(x_points.to(device), t_points.to(device))
        mse = torch.mean((preds - exact)**2).item()
    return mse

# ENtrenamos Superred
def train_pinn_darts(resample_every=100, n_interior=1000, n_initial=200, n_boundary=100,
                     switch_criterion='fixed', discrete_at=1000, stable_iters=100,
                     search_max_iterations=5000, post_switch_iterations=2000,
                     lr_weights_search=1e-3, lr_weights_compact=1e-4, lr_alpha=3e-4,
                     max_layers=7, neuron_options=None,
                     silent=True, disable_file_io=True, clip_norm=1.0, device=None,
                     return_search_only=False):
    if neuron_options is None:
        neuron_options = list(range(10, 151, 10))
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PINNDARTSNet(input_dim=2, output_dim=1, max_layers=max_layers, neuron_options=neuron_options).to(device)
    solver = AdvectionSolver()

    base = solver.generate_data(n_interior=n_interior, n_initial=n_initial, n_boundary=n_boundary, device=device)
    x_ic, t_ic, u_ic = base['initial']
    x_bc, t_bc, u_bc = base['boundary']

    def resample_interior(n, dev):
        x_int = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
        t_int = torch.rand(n, 1) * (solver.domain_t[1] - solver.domain_t[0])
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)
        return x_int.to(dev), t_int.to(dev)

    def sample_initial(n, dev):
        x_ic_s = torch.rand(n, 1) * (solver.domain_x[1] - solver.domain_x[0]) + solver.domain_x[0]
        t_ic_s = torch.zeros(n, 1)
        u_ic_s = solver.front(x_ic_s)
        x_ic_s.requires_grad_(True)
        t_ic_s.requires_grad_(True)
        return x_ic_s.to(dev), t_ic_s.to(dev), u_ic_s.to(dev)

    def sample_boundary(n, dev):
        x_bc_s = torch.ones(n, 1) * solver.domain_x[0]
        t_bc_s = torch.rand(n, 1) * (solver.domain_t[1] - solver.domain_t[0])
        u_bc_s = torch.ones(n, 1)
        x_bc_s.requires_grad_(True)
        t_bc_s.requires_grad_(True)
        return x_bc_s.to(dev), t_bc_s.to(dev), u_bc_s.to(dev)

    x_int_train, t_int_train = resample_interior(n_interior, device)
    train_data = {
        'interior': (x_int_train, t_int_train),
        'initial': (x_ic.to(device), t_ic.to(device), u_ic.to(device)),
        'boundary': (x_bc.to(device), t_bc.to(device), u_bc.to(device))
    }

    x_int_val, t_int_val = resample_interior(n_interior, device)
    x_ic_val, t_ic_val, u_ic_val = sample_initial(n_initial, device)
    x_bc_val, t_bc_val, u_bc_val = sample_boundary(n_boundary, device)
    val_data = {
        'interior': (x_int_val, t_int_val),
        'initial': (x_ic_val, t_ic_val, u_ic_val),
        'boundary': (x_bc_val, t_bc_val, u_bc_val)
    }

    optimizer_weights = torch.optim.Adam([p for n, p in model.named_parameters() if 'alpha' not in n], lr=lr_weights_search)
    optimizer_alpha = torch.optim.Adam(model.alpha_parameters, lr=lr_alpha)

    losses = []
    best_loss = float('inf')
    best_architecture = None
    final_model = model

    current_arch = model.get_discrete_architecture()
    stable_count = 0
    last_arch = current_arch

    start_time_search = time.time()
    switched_to_compact = False
    compact = None
    optimizer_compact = None
    switch_iteration = None
    comp_start_time = None
    architecture_search_time = None

    # MÉTRICAS que queremos registrar
    search_first_loss = None
    inherited_initial_loss = None
    compact_first_train_loss = None

    iteration = 0
    while True:
        # criterios de parada:
        if (not switched_to_compact) and (iteration >= search_max_iterations):
            break
        if switched_to_compact and (switch_iteration is not None) and (iteration >= switch_iteration + post_switch_iterations):
            break

        # Remuestreo
        if (iteration % resample_every) == 0 and iteration > 0 and (not switched_to_compact):
            x_int_train, t_int_train = resample_interior(n_interior, device)
            train_data['interior'] = (x_int_train, t_int_train)

        should_switch = False
        if not switched_to_compact:
            if switch_criterion == 'fixed' and iteration >= discrete_at:
                should_switch = True
                switch_iteration = iteration
            elif switch_criterion == 'stable':
                current_arch = model.get_discrete_architecture()
                if current_arch == last_arch:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_arch = current_arch
                if stable_count >= stable_iters:
                    should_switch = True
                    switch_iteration = iteration

        if (not switched_to_compact) and (not should_switch):
            model.train()

            #Actualizamos pesos
            optimizer_weights.zero_grad()
            total_loss_train, loss_pde_t, loss_ic_t, loss_bc_t = solver.compute_loss(model, train_data, discrete_architecture=False)

            if search_first_loss is None:
                search_first_loss = total_loss_train.item()

            total_loss_train.backward()
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if 'alpha' not in n], clip_norm)
            optimizer_weights.step()

            # Actualizamos pesos arquitectónicos
            optimizer_alpha.zero_grad()
            total_loss_val, loss_pde_v, loss_ic_v, loss_bc_v = solver.compute_loss(model, val_data, discrete_architecture=False)
            grads = torch.autograd.grad(total_loss_val, model.alpha_parameters, retain_graph=False)
            for p, g in zip(model.alpha_parameters, grads):
                if g is None:
                    p.grad = None
                else:
                    if p.grad is None:
                        p.grad = g.clone()
                    else:
                        p.grad.copy_(g)
            optimizer_alpha.step()

        else:
            if should_switch and not switched_to_compact:
                arch = model.get_discrete_architecture()
                architecture_search_time = time.time() - start_time_search

                if return_search_only:
                    return {
                        'searched_supernet': model,
                        'solver': solver,
                        'arch': arch,
                        'best_architecture': best_architecture,
                        'best_loss_search': best_loss,
                        'architecture_search_time': architecture_search_time,
                        'search_first_loss': search_first_loss,
                        'losses_search': losses.copy(),
                        'search_iterations_done': iteration,
                        'stable_stop': True
                    }

                compact = build_compact_and_transfer(model, input_dim=model.input_dim, output_dim=model.output_dim, verbose=False)
                compact = compact.to(device)

                with torch.no_grad():
                    compact.eval()
                    try:
                        inherited_loss, _, _, _ = solver.compute_loss_external(compact, train_data)
                        inherited_initial_loss = inherited_loss.item()
                    except Exception:
                        inherited_initial_loss = None

                switched_to_compact = True
                final_model = compact
                architecture_search_time = time.time() - start_time_search
                comp_start_time = time.time()

                optimizer_compact = torch.optim.Adam(compact.parameters(), lr=lr_weights_compact)

        if switched_to_compact:
            compact.train()
            optimizer_compact.zero_grad()
            total_loss, loss_pde, loss_ic, loss_bc = solver.compute_loss_external(compact, train_data)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(compact.parameters(), clip_norm)
            optimizer_compact.step()

            if compact_first_train_loss is None:
                compact_first_train_loss = total_loss.item()

        if switched_to_compact:
            losses.append(total_loss.item())
        else:
            losses.append(total_loss_train.item())

        current_total = losses[-1]
        if current_total < best_loss:
            best_loss = current_total
            best_architecture = model.get_discrete_architecture()
            if not disable_file_io:
                if switched_to_compact:
                    torch.save(compact.state_dict(), 'best_compact_inherited.pth')
                else:
                    torch.save(model.state_dict(), 'best_pinn_darts_model.pth')

        iteration += 1

    total_training_time = time.time() - start_time_search
    if return_search_only and not switched_to_compact:
        # La búsqueda llegó al tope de iteraciones sin estabilizarse: discretizamos igualmente.
        return {
            'searched_supernet': model,
            'solver': solver,
            'arch': model.get_discrete_architecture(),
            'best_architecture': best_architecture,
            'best_loss_search': best_loss,
            'architecture_search_time': total_training_time,
            'search_first_loss': search_first_loss,
            'losses_search': losses.copy(),
            'search_iterations_done': iteration,
            'stable_stop': False
        }
    if switched_to_compact and comp_start_time is not None:
        compact_training_time = time.time() - comp_start_time
    else:
        compact_training_time = 0.0
        architecture_search_time = total_training_time

    final_loss = losses[-1] if len(losses) > 0 else float('nan')

    metrics = {
        'search_first_loss': search_first_loss,
        'inherited_initial_loss': inherited_initial_loss,
        'compact_first_train_loss': compact_first_train_loss
    }

    return final_model, solver, losses, best_architecture, best_loss, final_loss, architecture_search_time, compact_training_time, switch_iteration, metrics
# Gráficas
def evaluate_and_plot(model_or_net, solver, best_architecture, final_loss, best_loss=None, title_suffix="", disable_file_io=True, save_plot=False, outdir="."):
    device = next(model_or_net.parameters()).device

    x_test = torch.linspace(-1, 1, 100).reshape(-1, 1).to(device)
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, t in enumerate(t_values):
        if i >= len(axes):
            break
        t_test = torch.ones_like(x_test) * t
        inputs = torch.cat([x_test, t_test], dim=1)
        with torch.no_grad():
            model_or_net.eval()
            prediction = model_or_net(inputs) if not isinstance(model_or_net, PINNDARTSNet) else model_or_net(inputs, discrete_architecture=True)
            exact = solver.exact_solution(x_test, t_test)
        axes[i].plot(x_test.cpu().numpy(), exact.cpu().numpy(), linewidth=2, label='Solución Exacta')
        axes[i].plot(x_test.cpu().numpy(), prediction.cpu().numpy(), linestyle='--', linewidth=2, label='Modelo')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('u(x,t)')
        axes[i].set_title(f't = {t}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        mse = torch.mean((prediction - exact)**2).item()
        axes[i].text(0.05, 0.95, f'MSE:{mse:.2e}', transform=axes[i].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].axis('off')
    arch_text = "Arquitectura Final:" + (str(best_architecture) if best_architecture is not None else "N/A")
    axes[-1].text(0.1, 0.9, arch_text, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[-1].text(0.1, 0.1, f'Pérdida final: {final_loss:.6f}', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    if save_plot and (not disable_file_io):
        outpath = os.path.join(outdir, f'pinn_darts_results{title_suffix}.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"  Figura guardada en {outpath}")
    plt.show()
    return fig

def aggregate_advection_metrics(seed_summaries):
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
