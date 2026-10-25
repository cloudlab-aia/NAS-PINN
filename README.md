<h1 align="center">Weight Inheritance in Differentiable Neural Architecture Search for Physics-Informed Neural Networks</h1>
<p align="center">Code for the experiments on neural architecture search, weight inheritance, and warm-up strategies in PINNs.</p>

This repository contains the code of the experiments reported in the paper **"Weight inheritance in differentiable neural architecture search for physics-informed neural networks: an empirical study"** (under review).

The project studies **weight inheritance in differentiable neural architecture search (NAS) for physics-informed neural networks (PINNs)**: whether the parameters learned in the continuous supernetwork during the search can be reused to initialize the final discretized architecture, instead of retraining it from scratch. The repository includes experiments on **advection** and **Poisson** benchmark problems, comparing **scratch initialization**, **weight inheritance**, and **inheritance with a short warm-up stage**.

## Contents

- **`advection_nas.py`**: the advection benchmark (ReLU): supernetwork, search, discretization, inheritance, scratch baseline and three-way final comparison.
- **`poisson_nas.py`**: the same components for the Poisson benchmark (tanh).
- **`PINN_HDARTS_v2.ipynb`**: driver notebook (four cells: imports, config, run all seeds, summary tables and figures).
- **`results/`**: per-seed summaries and loss curves (`*_summaries.json`, `*_losses.json`) of the runs reported in the paper, and the figures.

The code covers:
- the DARTS-style supernetwork (`PINNDARTSNet`, `MixedOperation`): each searchable layer holds one fully connected branch per candidate width plus an identity (skip) branch; branch outputs are zero-padded to the maximum width and mixed with softmax-weighted architecture parameters;
- the unsupervised search (`train_pinn_darts`), which updates network weights and architecture parameters with physics-informed losses and stops once the argmax architecture has been stable for a given number of iterations;
- discretization (`get_discrete_architecture`): the branch with the largest architecture parameter is kept in each layer;
- weight inheritance (`build_compact_and_transfer`): the discrete PINN is built with the selected branches and initialized with their supernetwork parameters, restricted to the columns that act on the previous selected width;
- the scratch baseline (`build_network_from_architecture`) with the same shape and random initialization;
- the three-way final comparison with optional warm-up (`run_*_final_comparison`, `run_*_seed_experiment`).

## Requirements

- Python >= 3.8
- PyTorch >= 1.10, NumPy >= 1.21, Matplotlib >= 3.4, Pandas >= 1.3
- Jupyter Notebook or JupyterLab
- Optional: CUDA-compatible GPU

## Installation and use

```bash
git clone https://github.com/cloudlab-aia/NAS-PINN.git
cd NAS-PINN
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook PINN_HDARTS_v2.ipynb
```

To reproduce the paper, run the four cells of the notebook in order. The RUN ALL cell runs seeds 0–4 of both benchmarks (about 2.5 h on a laptop GPU) and writes `results/`; the RESUMEN cell builds the tables and figures from `results/` and can be run while RUN ALL is in progress.

## Benchmarks

- **1D advection equation** $u_t + 1.2\,u_x = 0$ on $[-1,1]\times[0,1]$, candidate widths {25, 50, 75, 100}, ReLU.
- **2D Poisson equation** $-\Delta u = 2\pi^2 \sin(\pi x)\sin(\pi y)$ on $[0,1]^2$, candidate widths {50, 60, 70, 80}, tanh.

Evaluated strategies:
- **Scratch**: final discrete PINN trained from random initialization.
- **Inheritance**: compatible weights transferred from the searched supernetwork.
- **Inheritance + Warm-up**: inherited model trained with a short low-learning-rate warm-up before standard final training.

## Data

All training data are synthetic: collocation points sampled in the domain and boundary/initial points evaluated from the analytical expressions. No external datasets are required.

## Acknowledgements

This research has been performed for the research project <a href="https://aia.ua.es/en/proyectos/federated-serverless-architectures-for-heterogeneous-high-performance-computing-in-smart-manufacturing.html" target="_blank">Federated Serverless Architectures for Heterogeneous High Performance Computing in Smart Manufacturing</a>, at the Applied Intelligent Architectures Research Group of the University of Alicante (Spain).

Grant <b>Serverless4HPC PID2023-152804OB-I00</b> funded by MICIU/AEI/10.13039/501100011033 and by ERDF/EU.

## Citation

```bibtex
@misc{naspinn_repo,
  title        = {Weight inheritance in differentiable neural architecture search for physics-informed neural networks: an empirical study},
  author       = {Mu\~{n}oz Hern\'{a}ndez, David and Herrero G\'{o}mez, Pablo and Garc\'{i}a Gelardo, Alejandro and Jimeno-Morenilla, Antonio and Mora, Higinio},
  year         = {2026},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/cloudlab-aia/NAS-PINN}}
}
```

## License

GPL-3.0. See `LICENSE`.
