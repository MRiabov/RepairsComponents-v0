# Use the CUDA-ready base image from Vast.ai
FROM vastai/base-image:cuda-12.4.1-auto

# Make sure bash is the default shell for subsequent commands
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1. System-level dependencies (mirrors dev/setup.sh)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-dev \
        libsparsehash-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Python build tool
RUN pip install --no-cache-dir uv

# 3. Copy project into image
WORKDIR /workspace/RepairsComponents-v0
COPY . /workspace/RepairsComponents-v0

# 4. Create a local virtual-environment and reproduce the Python parts
RUN uv venv && \
    source .venv/bin/activate && \
    # Install uv inside the venv for subsequent package installs
    pip install --no-cache-dir uv && \
    # Main requirements (equivalent to combined_req.txt)
    uv pip install -r /workspace/RepairsComponents-v0/combined_req.txt -U && \
    # Additional packages installed explicitly in setup.sh
    uv pip install build123d==0.9.1 torch==2.5.1 torchvision setuptools && \
    uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html && \
    # Constrain numpy and install the library in editable mode
    uv pip install numpy==1.26.4 --no-deps && \
    uv pip install -e /workspace/RepairsComponents-v0/. --no-deps && \
    # Install torchsparse directly from GitHub
    uv pip install git+https://github.com/mit-han-lab/torchsparse --no-build-isolation

# 5. (Optional) Git identity copied from setup.sh (harmless inside container)
RUN git config --global user.name "MRiabov" && \
    git config --global user.email "maksymriabov2004@gmail.com"

# 6. Activate the venv for every container session
ENV PATH="/workspace/RepairsComponents-v0/.venv/bin:${PATH}"

CMD ["bash"]
