FROM nvcr.io/nvidia/pytorch:25.08-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=ubuntu

# 1. System dependencies
# Added 'ninja-build' which is critical for fast CUDA extension compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sudo \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 2. User Setup (Robust)
# We check if the user exists; if not, we create it.
# If the group exists, we reuse it; otherwise, we create it.
RUN if getent group $GROUP_ID; then \
        groupmod -n $USER_NAME $(getent group $GROUP_ID | cut -d: -f1); \
    else \
        groupadd -g $GROUP_ID $USER_NAME; \
    fi && \
    if id -u $USER_ID >/dev/null 2>&1; then \
        usermod -l $USER_NAME -g $GROUP_ID -d /home/$USER_NAME -m $(getent passwd $USER_ID | cut -d: -f1); \
    else \
        useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USER_NAME; \
    fi && \
    usermod -aG sudo $USER_NAME && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /workspace
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
# Fix for common header issues in geometric extensions
ENV MAX_JOBS=4

# 4. Python Fundamentals
# Install numpy 1.x FIRST, as subsequent compilations (diso/cumesh) may need numpy headers.
RUN pip install --upgrade pip && \
    pip install "numpy<2.0" ninja

# 5. Compiled Extensions (Layered for caching)
# We separate these so if one fails, you don't rebuild them all.
RUN pip install --no-build-isolation git+https://github.com/eliphatfs/cumesh2sdf.git

RUN pip install --no-build-isolation git+https://github.com/seonghunn/pdmc.git

# diso is sensitive to CUDA versions, ensuring it runs isolated
RUN FORCE_CUDA=1 pip install --no-build-isolation diso==0.1.4

# 6. Standard Libs
RUN pip install libigl==2.5.1 trimesh==4.4.*

# 7. Local Source Setup
# We change ownership explicitly during the copy to avoid permission issues
COPY --chown=$USER_NAME:$USER_NAME . /workspace/pamo

WORKDIR /workspace/pamo

# 8. Install Local Packages
# Using --no-deps is often safer here if you've already handled dependencies above, 
# but we'll stick to standard install to catch missing ones.
RUN cd simp_cuda && \
    FORCE_CUDA=1 pip install --no-build-isolation .

RUN cd simp_cuda/safe_project && \
    FORCE_CUDA=1 pip install --no-build-isolation .

# 9. Final Configuration
# DANGEROUS: ENV LD_PRELOAD...
# Instead of setting this globally (which breaks system tools), 
# we rely on the path precedence or set it strictly for the entrypoint if absolutely necessary.
# If libigl fails without it, prefer setting it in a shell alias or specific run command.

USER $USER_NAME

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]