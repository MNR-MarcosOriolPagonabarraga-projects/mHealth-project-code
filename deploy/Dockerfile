FROM ubuntu:22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies, ARM cross-compiler, and QEMU emulator
RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    git \
    build-essential \
    gcc-arm-none-eabi \
    qemu-system-arm \
    && rm -rf /var/lib/apt/lists/*

# Install Zig (Latest Stable)
RUN wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz && \
    tar -xf zig-linux-x86_64-0.11.0.tar.xz && \
    mv zig-linux-x86_64-0.11.0 /usr/local/zig && \
    rm zig-linux-x86_64-0.11.0.tar.xz

ENV PATH="/usr/local/zig:${PATH}"

# Install onnx2c to convert our ONNX model to a pure C header
RUN git clone https://github.com/kraiskil/onnx2c.git /opt/onnx2c && \
    cd /opt/onnx2c && \
    make

ENV PATH="/opt/onnx2c:${PATH}"

WORKDIR /workspace
CMD ["bash"]