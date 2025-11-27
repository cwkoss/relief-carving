# ZoeDepth Docker Environment with Compatible Versions
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with exact versions for compatibility
RUN pip install --no-cache-dir \
    timm==0.6.12 \
    opencv-python==4.6.0.66 \
    matplotlib==3.6.2 \
    h5py==3.7.0 \
    scipy==1.10.0 \
    tqdm==4.64.1 \
    huggingface-hub==0.11.1 \
    wandb==0.13.9

# Copy the ZoeDepth repository
COPY . /app/zoedepth

# Set the working directory to ZoeDepth
WORKDIR /app/zoedepth

# Download MiDaS dependency (this helps avoid download issues later)
RUN python -c "import torch; torch.hub.help('intel-isl/MiDaS', 'DPT_BEiT_L_384', force_reload=True)" || true

# Create a simple test script
RUN echo 'import warnings\n\
warnings.filterwarnings("ignore")\n\
import torch\n\
print("Testing ZoeDepth in Docker...")\n\
print(f"PyTorch version: {torch.__version__}")\n\
print(f"CUDA available: {torch.cuda.is_available()}")\n\
try:\n\
    from zoedepth.models.builder import build_model\n\
    from zoedepth.utils.config import get_config\n\
    print("[OK] ZoeDepth modules imported successfully")\n\
    \n\
    # Test model loading\n\
    model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)\n\
    print("[OK] ZoeD_N model loaded successfully!")\n\
    print("SUCCESS: ZoeDepth is working in Docker!")\n\
except Exception as e:\n\
    print(f"[ERROR] {e}")\n\
' > test_docker.py

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "test_docker.py"]