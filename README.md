<h1 align="center">Advances in Neural Architecture Search for Physics-Informed Neural Networks</h1>
<p align="center">Code repository for the experiments on neural architecture search, weight inheritance, and warm-up strategies in PINNs.</p>

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

- Python >= 3.8
- [PyTorch](https://pytorch.org/) >= 1.10
- NumPy >= 1.21
- Matplotlib >= 3.4
- Pandas >= 1.3
- Jupyter Notebook or JupyterLab
- Optionally: CUDA-compatible GPU for faster training

## Installation and use

Follow these steps to set up and run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USER-OR-ORG/NAS-PINN.git
cd NAS-PINN
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open:

```text
nas_pinn_experiments.ipynb
```

## Benchmarks

The experiments in this repository include:

- **1D advection equation**
- **2D Poisson equation**

The evaluated strategies are:

- **Scratch**: final discrete PINN trained from random initialization
- **Inheritance**: compatible weights transferred from the searched supernetwork
- **Inheritance + Warm-up**: inherited model trained with a short low-learning-rate warm-up before standard final training

## Data

The models in this project are trained on synthetic data generated from the analytical expressions of the PDE benchmarks and from collocation-point sampling. No external datasets are required.

## Repository status

This repository is intended to accompany the paper and may be updated with cleaned scripts, additional documentation, and reproducibility material.

## Acknowledgements

This research has been performed for the research project <a href="https://aia.ua.es/en/proyectos/federated-serverless-architectures-for-heterogeneous-high-performance-computing-in-smart-manufacturing.html" target="_blank">Federated Serverless Architectures for Heterogeneous High Performance Computing in Smart Manufacturing</a>, at the Applied Intelligent Architectures Research Group of the University of Alicante (Spain).

Grant <b>Serverless4HPC PID2023-152804OB-I00</b> funded by MICIU/AEI/10.13039/501100011033 and by ERDF/EU.

## Citation

If you use this repository, please cite the corresponding paper.

```bibtex
@misc{naspinn_repo,
  title        = {Advances in Neural Architecture Search for Physics-Informed Neural Networks},
  author       = {David Muñoz Hernández and Pablo Herrero Gómez and Alejandro García Gelardo and Antonio Jimeno-Morenilla and Higinio Mora},
  year         = {2025},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/YOUR-USER-OR-ORG/NAS-PINN}}
}
```

If the paper is accepted or uploaded as a preprint, replace this entry with the final publication citation.

## License

This project is licensed under the GPL-3.0 License. See the `LICENSE` file for details.
