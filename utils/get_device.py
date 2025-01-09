def get_device():
    from torch import cuda, tensor

    try:
        device = "cpu"
        if cuda.is_available():
            x = tensor([0.0], device="cuda")

            if "cuda" in x.device.type:
                device = "cuda"
    except:
        device = "cpu"

    return device