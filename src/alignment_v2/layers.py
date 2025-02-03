from torch import nn

REGISTRY_REQUIREMENTS = ["name", "layer_index", "unfold", "ignore", "flag"]

LAYER_REGISTRY = {
    nn.Linear: {"name": "linear", "layer_index": None, "unfold": False, "ignore": False, "flag": False},
    nn.Conv2d: {"name": "conv2d", "layer_index": None, "unfold": True, "ignore": False, "flag": True},
}

def default_metaprms_ignore(name):
    return {"name": name, "layer_index": None, "unfold": False, "ignore": True, "flag": True}

def default_metaprms_linear(index, name="linear", flag=False):
    return {"name": name, "layer_index": index, "unfold": False, "ignore": False, "flag": flag}

def default_metaprms_conv2d(index, name="conv2d", flag=True):
    return {"name": name, "layer_index": index, "unfold": True, "ignore": False, "flag": flag}

def check_metaparameters(metaparameters, throw=True):
    for key in REGISTRY_REQUIREMENTS:
        if key not in metaparameters:
            if throw:
                raise ValueError(f"Missing key: {key}")
            return False
    return True

for lt, mp in LAYER_REGISTRY.items():
    check_metaparameters(mp)