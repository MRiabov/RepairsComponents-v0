# RepairsComponents-v0

The first library enabling training of robotics on Repair and Maintenance problems - mechanics and electronics included!

## Overview

Repairs Components is a library designed to enable manufacturing companies to assemble and repair their products using robotics. Simply put - with a few manipulations on your STEP assembly file, this library allows training of a robot to assemble, disassemble and replace parts in your assembly. You can bring your own robot!

## Capabilities

- **A RL environment to assemble and disasseble mechanical assemblies**: Upload a `.step` assembly file and let Repairs paralel environment do the work for you.
- **Replace a damaged component**: A common case for maintenance is a damaged component. This RL environment teaches a robot to replace various components under *thousands* of various settings *every second*.
- **Multiple modes of training** - Our library can be used as a dataset generator for offline reinforcement learning. Simply flip "save" to True (under io_cfg config dict) in RL and video, voxel, and data observation will be persisted to your disk

- **Electronics**: 
  - Components such as buttons and switches, allowing interaction with the environment; wires, connectors allowing the robots to assemble a given env. _(WIP)_
 
Note: this is a *reinforcement learning environment*. You will need a *reinforcement learning algorithm* and we provide that in [Repairs-v0](https://github.com/MRiabov/Repairs-v0)

> [!WARNING]  
> Caution: the electronics library is very much in Alpha and active development and could be broken when you see this. Check in a few weeks when we will polish it!

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
