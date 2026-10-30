ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.description="FastFlowLM build environment with all dependencies pre-installed"
LABEL org.opencontainers.image.source="https://github.com/ROCm/FastFlowLM"

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_PPA=""
ARG BACKPORTS=""

# Set up PPA if needed
RUN if [ -n "$UBUNTU_PPA" ]; then \
        apt update && apt install -y software-properties-common && \
        add-apt-repository -y "$UBUNTU_PPA"; \
    fi

# setup backports if needed
RUN if [ -n "$BACKPORTS" ]; then \
        echo "deb http://deb.debian.org/debian $BACKPORTS main" >> /etc/apt/sources.list; \
        apt update; \
        apt install -t $BACKPORTS -y libxrt-dev; \
    fi

# Install all build dependencies
RUN apt update && apt install -y \
    build-essential \
    ca-certificates \
    cargo \
    ccache \
    cmake \
    curl \
    debhelper-compat \
    dpkg-dev \
    fakeroot \
    git \
    libboost-dev \
    libboost-program-options-dev \
    libcurl4-openssl-dev \
    libdrm-dev \
    libedit-dev \
    libxrt-dev \
    nasm \
    ninja-build \
    patchelf \
    pkg-config \
    rustc \
    uuid-dev \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# The distro's rustc is too old for the tokenizers crate, so CI used to
# install stable with rustup in every job. Baked in here instead, ahead of the
# distro toolchain on PATH (whose cargo/rustc packages stay: the .deb's
# Build-Depends name them). World-writable so a job running as any user can
# update the registry.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path \
    && chmod -R a+w "$RUSTUP_HOME" "$CARGO_HOME" \
    && rustc --version

WORKDIR /workspace

CMD ["/bin/bash"]
