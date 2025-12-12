FROM nvcr.io/nvidia/pytorch:25.08-py3

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Set CUDA architectures for Blackwell (RTX 50 series) support
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0"

# Python dependencies (CUDA extensions require FORCE_CUDA=1 and --no-build-isolation)
RUN pip install --upgrade pip && \
    pip install warp-lang --pre --extra-index-url https://pypi.nvidia.com && \
    pip install --no-build-isolation git+https://github.com/eliphatfs/cumesh2sdf.git && \
    FORCE_CUDA=1 pip install --no-build-isolation git+https://github.com/seonghunn/pdmc.git && \
    FORCE_CUDA=1 pip install --no-build-isolation diso==0.1.4 && \
    pip install libigl==2.5.1 numpy==1.* trimesh==4.4.*

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Copy local pamo source (with fixes applied)
COPY . /workspace/pamo

WORKDIR /workspace/pamo

# Install simp_cuda (stage 2: simplification)
RUN cd simp_cuda && FORCE_CUDA=1 pip install --no-build-isolation .

# Install safe_project (stage 3: safe projection)
RUN cd simp_cuda/safe_project && FORCE_CUDA=1 pip install --no-build-isolation .

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]
