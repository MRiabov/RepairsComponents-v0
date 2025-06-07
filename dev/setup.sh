pip install uv
cd RepairsComponents-v0/
uv venv
source .venv/bin/activate
uv pip install -r combined_req.txt -U
uv pip install build123d==0.9.1 torch torchvision -U
uv pip install numpy==1.26.4 --no-deps # note: I got this working by using not uv but standard pip.
uv pip install -e /workspace/RepairsComponents-v0/.  --no-deps
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
#note^ kaolin is pip installed, not uv.

# set venv as python interpreter in Windsurf if you need to debug

sudo apt install libgl1-mesa-dev -y
#else it won't install 

#and also install ruff as extension.

git config --global user.name "MRiabov"
git config --global user.email "maksymriabov2004@gmail.com"
