from torch import nn
from alignment import utils

# The LAYER_REGISTRY contains meta parameters for each type of layer used in alignment networks
REGISTRY_REQUIREMENTS = [
    "name",
    "layer_index",
    "unfold",
    "ignore",
    "flag",
]

LAYER_REGISTRY = {
    nn.Linear: {
        "name": "linear",
        "layer_index": None,
        "unfold": False,
        "ignore": False,
        "flag": False,
    },
    nn.Conv2d: {
        "name": "conv2d",
        "layer_index": None,
        "unfold": True,
        "ignore": False,
        "flag": True,
    },
}

def default_metaprms_ignore(name):
    metaparameters = {
        "name": name,
        "layer_index": None,
        "unfold": False,
        "ignore": True,
        "flag": True,
    }
    return metaparameters

def default_metaprms_linear(index, name="linear", flag=False):
    metaparameters = {
        "name": name,
        "layer_index": index,
        "unfold": False,
        "ignore": False,
        "flag": flag,
    }
    return metaparameters

def default_metaprms_conv2d(index, name="conv2d", flag=True):
    metaparameters = {
        "name": name,
        "layer_index": index,
        "unfold": True,
        "ignore": False,
        "flag": flag,
    }
    return metaparameters

def check_metaparameters(metaparameters, throw=True):
    if not all([required in metaparameters for required in REGISTRY_REQUIREMENTS]):
        if throw:
            raise ValueError(f"metaparameters are missing required keys, it requires all of the following: {REGISTRY_REQUIREMENTS}")
        return False
    return True

for layer_type, metaparameters in LAYER_REGISTRY.items():
    if not check_metaparameters(metaparameters, throw=False):
        raise ValueError(
            f"Layer type: {layer_type} from the `LAYER_REGISTRY` is missing metaparameters. "
            f"It requires all of the following: {REGISTRY_REQUIREMENTS}"
        )