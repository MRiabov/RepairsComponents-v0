# RepairsComponents-v0
The first library enabling training of robotics on Repair and Maintenance problems - mechanics and electronics included!

## Overview 

Repairs Components is a library designed to enable manufacturing companies to assemble and repair their products using robotics. Simply put - with a few manipulations on your STEP assembly file, this library allows training of a robot to assemble, disassemble and replace parts in your assembly. You can bring your own robot!

## Capabilities⚡

- **Teach a robot to assemble your product**: If you want to replace manual workers with robotic arms, this environment is a "batteries-included" simulator for it's training:
  - Complex mechanical and realistic assemblies - we support assemblies of up to 30 parts with using fasteners.
  - Realistic visuals - set realistic textures for your product, or choose random ones for a more thorough (generalizable) training.
  - Multiple ways to train your model: we support training with dense rewards (frequent rewards on completion of a basic task) and sparse rewards when the reward is given in the end.
  - Multiple ways to observe your environment: you can choose what to feed into your model: camera observations, 3d voxel shapes, or even graphs of joined/disjoined with fasteners assemblies. Electronics can be input as a graph of components.
  - Domain randomization: create thousands of variations of disassembled components and train your robot to assemble from any starting point.
  - Minimum setup: all the setup is already done for you - plug your product in and let it run.
- **Replace a damaged component**: A common case for maintenance is a damaged component. This RL environment teaches a robot to replace various components under *millions* of different settings.
- **Disassemble your product** - for e.g. recyling/repair tasks, a disassembly mode was created. 
- **Offline dataset generation** - All of the above can be used as a offline reinforcement learning dataset generator for Vision Transformers or standard offline RL. Simply flip "save" to True (under io_cfg config dict) and video, voxel, and graph observations will be persisted to your disk.
- **Electronics**: Components such as buttons and switches, allowing interaction with the environment; wires, standard connectors (e.g. XT-60/USB-A/USB-C) allowing the robots to assemble a given assembly. _(Work in progress)_
 
Note: this is a *reinforcement learning environment*. You will need a *reinforcement learning algorithm* and we provide that in [Repairs-v0](https://github.com/MRiabov/Repairs-v0). You should also try using [Vision-Language-Action models](https://arxiv.org/abs/2406.09246).

> [!WARNING]  
> Caution: the electronics library is very much in Alpha and active development and could be broken when you see this. Check in a few weeks when we will polish it!

## Installation 🛠️

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

## Roadmap 🛣️
1. Add welding.
2. Add realistic wires and their constraints.
3. Add more electronics components, and support assembly of electromechanical (motor) assemblies.
4. Add softbody support (e.g. for automotive hoses).
