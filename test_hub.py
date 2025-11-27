import torch
import warnings

print("PyTorch version:", torch.__version__)

try:
    # Try loading ZoeDepth model via torch hub
    print("Loading ZoeDepth model...")

    # First load MiDaS to ensure the dependency is available
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    print("MiDaS loaded successfully")

    # Try to load ZoeDepth model
    model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True)
    print("ZoeD_N model loaded successfully!")
    print("Model type:", type(model_zoe_n))

except Exception as e:
    print("Error loading model:", e)