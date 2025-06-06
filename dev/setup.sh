pip install uv
uv pip install -r combined_req.txt -U
uv pip install build123d==0.9.1 
uv pip install numpy==1.26.4 --no-deps # note: I got this working by using not uv but standard pip.
uv pip install -e /workspace/RepairsComponents-v0/.  


sudo apt install libgl1-mesa-dev
#else it won't install 

#and also install ruff as extension.

git config --global user.name "MRiabov"
git config --global user.email "maksymriabov2004@gmail.com"
