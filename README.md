<h1 align="center">Advances in Neural Architecture Search for Physics-Informed Neural Networks</h1>
<p align="center">Code repository for the experiments on NAS, weight inheritance, and warm-up strategies in PINNs.</p>

<p align="center">
  <a href="PUT_PAPER_LINK_HERE">
    <img src="https://img.shields.io/badge/Neurocomputing-Under%20Review-orange" alt="Paper Status">
  </a>
</p>

This repository contains the code implementation of the experiments performed for the paper **"Advances in Neural Architecture Search for Physics-Informed Neural Networks"**.

The project studies **weight inheritance in differentiable neural architecture search (NAS) for physics-informed neural networks (PINNs)**, focusing on whether parameters learned in the continuous searched supernetwork can be reused effectively in the final discretized architecture. The repository includes experiments on **advection** and **Poisson** benchmark problems, together with comparisons between **scratch initialization**, **weight inheritance**, and **inheritance with a short warm-up stage**.

## Contents

The repository currently includes:

- **`nas_pinn_experiments.ipynb`**: main Jupyter notebook containing the experiments and analysis used in the paper.
- **`figures/`**: representative convergence plots used in the manuscript.
- **`requirements.txt`**: Python dependencies required to run the notebook.

The code covers:
- differentiable NAS for PINNs,
- discretization of the searched supernetwork,
- transfer of compatible parameters to the final discrete PINN,
- optional warm-up adaptation,
- evaluation on advection and Poisson PDE benchmarks.

## Requirements

To run this project, you need:

- Python ≥ 3.8
- [PyTorch](https://pytorch.org/) ≥ 1.10
- NumPy ≥ 1.21
- Matplotlib ≥ 3.4
- Pandas ≥ 1.3
- Jupyter Notebook or JupyterLab
- Optionally: CUDA-compatible GPU for faster training

## Installation and use

Follow these steps to set up and run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USER-OR-ORG/NAS-PINN.git
cd NAS-PINN
