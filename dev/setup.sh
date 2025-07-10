# git clone https://github.com/MRIabov/RepairsComponents-v0.git
# git clone https://github.com/MRIabov/Repairs-v0.git
sudo apt install libgl1-mesa-dev libsparsehash-dev -y

pip install uv
cd RepairsComponents-v0/
uv venv
source .venv/bin/activate
pip install uv
uv pip install -r /workspace/RepairsComponents-v0/combined_req.txt -U # faster install of torchsparse
uv pip install torch==2.5.1 torchvision setuptools git+https://github.com/gumyr/build123d
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html


uv pip install numpy==1.26.4 --no-deps # note: I got this working by using not uv but standard pip.
uv pip install -e /workspace/RepairsComponents-v0/.  --no-deps
# pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
# #note^ kaolin is pip installed, not uv.
# uv pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

uv pip install git+https://github.com/mit-han-lab/torchsparse --no-build-isolation

# set venv as python interpreter in Windsurf if you need to debug

#else it won't install 

#and also install ruff as extension.

git config --global user.name "MRiabov"
git config --global user.email "maksymriabov2004@gmail.com"

# to run:
# uv run /workspace/Repairs-v0/neural_nets/sac_repairs.py
# note: if pyright or windsurf timeout, check whether default paths were added to pyright ignore.