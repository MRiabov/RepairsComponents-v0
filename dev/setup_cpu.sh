# git clone https://github.com/MRIabov/RepairsComponents-v0.git
# git clone https://github.com/MRIabov/Repairs-v0.git
export FORCE_CPU=1
sudo apt install libgl1-mesa-dev libsparsehash-dev -y

pip install uv
# cd RepairsComponents-v0/
uv venv
source .venv/bin/activate
python3 -m ensurepip & python3 -m pip install uv
uv pip install -r ~/Work/RepairsComponents-v0/combined_req_cpu.txt -U 
uv pip install torch==2.5.1 torchvision setuptools git+https://github.com/gumyr/build123d ninja --extra-index-url https://download.pytorch.org/whl/cpu
# uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html


uv pip install numpy==1.26.4 --no-deps # note: I got this working by using not uv but standard pip.
uv pip install -e ~/Work/RepairsComponents-v0/.  --no-deps
# pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
# #note^ kaolin is pip installed, not uv.
# uv pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

# uv pip install git+https://github.com/mit-han-lab/torchsparse --no-build-isolation
uv pip install 
git clone https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
TORCH_CUDA_ARCH_LIST="" FORCE_ONLY_CPU=1 uv pip install -v . --no-build-isolation

# set venv as python interpreter in Windsurf if you need to debug

#else it won't install 

#and also install ruff as extension.

git config --global user.name "MRiabov"
git config --global user.email "maksymriabov2004@gmail.com"

# to run:
# uv run ~/Work/Repairs-v0/neural_nets/sac_repairs.py
# note: if pyright or windsurf timeout, check whether default paths were added to pyright ignore.

#run tests:
# uv run pytest #folder#
# e.g. uv run pytest ~/Work/RepairsComponents-v0/tests/test_tool_genesis.py::test_attach_and_detach_tool_to_arm_with_fastener