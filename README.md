# RepairsComponents-v0

A modular library of physics-based engineering components for training on repair and maintenance simulations.

## Overview

Repairs Components is a library designed to provide a toolkit of reusable, physics-based engineering components for training on repair and maintenance simulations. The library is specifically tailored for use with reinforcement learning environments, particularly those built on the Genesis physics engine. 

## Features

- **Fasteners**: 
  - Fasteners between parts, allowing fastening of parts during simulation

- **Electronics**: 
  - Components such as buttons and switches, allowing interaction with the environment; wires, connectors allowing the robots to assemble a given env

## Installation

```sh
git clone https://github.com/MRIabov/RepairsComponents-v0.git
git clone https://github.com/MRIabov/Repairs-v0.git
sudo apt install libgl1-mesa-dev libsparsehash-dev -y

pip install uv
cd RepairsComponents-v0/
uv venv
source .venv/bin/activate
pip install uv
uv pip install -r /workspace/RepairsComponents-v0/combined_req.txt -U 
uv pip install build123d==0.9.1 torch==2.5.1 torchvision setuptools 
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
uv pip install numpy==1.26.4 --no-deps 
uv pip install -e /workspace/RepairsComponents-v0/.  --no-deps
uv pip install git+https://github.com/mit-han-lab/torchsparse --no-build-isolation
```
And run via `/workspace/.venv/bin/python /workspace/Repairs-v0/neural_nets/sac_repairs_torch.py` (adjusting for your venv path)
