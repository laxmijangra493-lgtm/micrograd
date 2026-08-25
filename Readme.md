<img width="564" height="564" alt="image" src="https://github.com/user-attachments/assets/7b824c2a-5ef5-4c1a-9c0e-7453ae9b3bf3" />


# Micrograd

A lightweight, scalar-valued autograd engine with a PyTorch-like Neural Network API built on top.

This project implements backpropagation over a dynamically constructed Directed Acyclic Graph (DAG) for educational and deep learning foundational research.

---

## Features

- **Autograd Engine**: Automatic differentiation tracking scalar operations (`+`, `-`, `*`, `/`, `**`, `relu`).
- **Neural Network API**: Modular components (`Neuron`, `Layer`, `MLP`) mirroring PyTorch syntax.
- **DAG Visualization**: Built-in Graphviz utility to visualize forward/backward computation graphs.
- **Zero External Dependencies**: Pure Python implementation with minimal overhead.

---

## Installation

Clone the repository and install requirements:

```bash
git clone [https://github.com/laxmijangra493-lgtm/micrograd.git](https://github.com/laxmijangra493-lgtm/micrograd.git)
cd micrograd
pip install -r requirements.txt


(Note: Install graphviz system package if you plan to render computation graphs also their was no requirements.txt file do it by yourself😊).



Repository Architecture:
micrograd/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py     # Core autograd computation engine
│   ├── nn.py         # Neuron, Layer, and MLP modules
│   └── visualize.py  # Graphviz DAG generator
├── test_engine.py    # PyTorch verification tests
├── requirements.txt
└── README.md


```
# This is for learning and trying new stuff's if you wanted to get the real SLM.py file and other related to mySLM you can head to my Repo named mesaggeGPT😊
