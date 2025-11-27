import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

try:
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
    print("ZoeDepth imports successful")

    # Try to build a model without pretrained weights first
    conf = get_config("zoedepth", "infer")
    conf['pretrained_resource'] = ""  # Disable pretrained loading
    model = build_model(conf)
    print("Model created successfully (without pretrained weights)")
    print("Model type:", type(model))

except Exception as e:
    print("Error:", e)