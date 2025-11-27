# ZoeDepth Docker Setup Instructions

## Prerequisites
1. Install Docker Desktop for Windows from: https://docs.docker.com/desktop/install/windows/
2. Ensure Docker Desktop is running (check system tray)
3. Enable WSL 2 if prompted during installation

## Quick Start (After Docker Installation)

### Option 1: Using the Batch File (Easiest)
```bash
# Double-click run_docker.bat or run in command prompt:
run_docker.bat
```

### Option 2: Using Docker Commands
```bash
# Build the Docker image
docker build -t zoedepth:latest .

# Run the container with volume mounts
docker run -it --rm \
    -v "%cd%":/app/zoedepth \
    -v "%cd%\images":/app/images \
    -v "%cd%\output":/app/output \
    -p 8080:8080 \
    zoedepth:latest
```

### Option 3: Using Docker Compose
```bash
# Start the service
docker-compose up --build

# To run interactively
docker-compose run --rm zoedepth bash
```

## What the Docker Setup Includes

1. **Compatible Base Image**: PyTorch 1.13.1 with CUDA 11.6
2. **Exact Package Versions**: All packages pinned to compatible versions
3. **Volume Mounts**:
   - Current directory → `/app/zoedepth` (code)
   - `images/` → `/app/images` (input images)
   - `output/` → `/app/output` (results)
4. **Port Mapping**: Port 8080 for potential web interfaces

## Usage Examples

### Basic Depth Estimation
```bash
# After Docker container is running:
python sanity.py  # Run the built-in test

# Or use the Python API:
python -c "
import torch
from PIL import Image
model = torch.hub.load('.', 'ZoeD_N', source='local', pretrained=True)
image = Image.open('/app/images/your_image.jpg').convert('RGB')
depth = model.infer_pil(image)
print(f'Depth estimation complete! Shape: {depth.shape}')
"
```

### Interactive Mode
```bash
# Run container in interactive mode:
docker run -it --rm -v "%cd%":/app/zoedepth zoedepth:latest bash

# Then inside container:
python
>>> import torch
>>> model = torch.hub.load('.', 'ZoeD_N', source='local', pretrained=True)
>>> # Your ZoeDepth code here...
```

## Troubleshooting

1. **Docker not found**: Install Docker Desktop and restart your terminal
2. **Permission errors**: Make sure Docker Desktop is running
3. **Build fails**: Ensure you're in the zoedepth directory with Dockerfile
4. **CUDA not available**: The container will run on CPU by default (still works)

## Next Steps After Docker Installation

1. Place test images in the `images/` folder
2. Run `run_docker.bat` or the docker commands above
3. Results will appear in the `output/` folder
4. The container includes a test script that verifies everything is working

## File Structure
```
zoedepth/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Compose configuration
├── run_docker.bat          # Windows batch script
├── DOCKER_SETUP.md         # This file
├── images/                 # Put your input images here
└── output/                 # Results will appear here
```