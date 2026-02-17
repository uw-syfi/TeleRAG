ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

# Set environment variables (best practice)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# --- FIX: Install Python and pip ---
# This ensures pip is available for subsequent steps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        curl \
        build-essential \
        git \
        libnuma-dev \
        numactl \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Use 'python3' command explicitly from now on
RUN python3 -m pip install --upgrade pip
# -----------------------------------

# Set the working directory
WORKDIR /app

# Install dependencies (Best practice for caching):
# 1. Copy only the requirements file first.
COPY requirements.txt .

# 2. Install dependencies. Using --no-cache-dir speeds things up and saves space.
RUN pip install --upgrade pip
RUN python3 -m pip install git+https://github.com/ozeliger/pyairports
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/
RUN pip install faiss-gpu-cu12==1.12.0

# Copy the rest of the application code into the container.
# This should include your experiment scripts, data, and configuration.
COPY . .
RUN git submodule update --init --recursive
COPY 3rdparty/sglang /repos/sglang
RUN cd /repos/sglang && pip install -e "python[all]"
COPY . /repos/ragacc
RUN cd /repos/ragacc && pip install -e .

# The environment is now ready.
# You will use 'docker exec' to install dependencies/clone repos inside it.

# Set a placeholder command to keep the container running indefinitely
CMD ["sleep", "infinity"]
