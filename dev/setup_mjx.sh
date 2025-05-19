# in one terminal:
# lib install
# sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev
cd ..
pip install -e .
pip install jax[cuda] flax genesis-world
# playground

#in second terminal (madrona install):
git clone https://github.com/shacklettbp/madrona_mjx.git
cd madrona_mjx
git submodule update --init --recursive
mkdir build
cd build
pip install cmake>=4
hash -r #this seems to restart the terminal? either way, it discovers pip's cmake.

cmake ..
make -j 3 # increase if the processor is good.

#madrona
cd ..
pip install -e .
