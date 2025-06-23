# External Module

This module contains third-party code and external dependencies that are integrated into the alignment framework.

## BROJA_2PID

### Overview

BROJA_2PID (Bivariate Partial Information Decomposition) is an external implementation of the BROJA estimator for computing partial information decomposition. It's maintained as external code because:

1. **Original Implementation**: It's a direct port of the original BROJA algorithm
2. **Academic Attribution**: Preserves the original authors' implementation
3. **License Compatibility**: May have different licensing terms
4. **Minimal Modifications**: Kept close to the original for accuracy

### Usage

The BROJA_2PID module is used internally by our PID (Partial Information Decomposition) metrics:

```python
from alignment.metrics import get_metric

# These metrics use BROJA_2PID internally
shared_info = get_metric('pid_shared')
unique_x = get_metric('pid_unique_x')
unique_y = get_metric('pid_unique_y')
synergy = get_metric('pid_synergy')

# Compute PID
scores_shared = shared_info.compute(X1, X2, Y)
scores_unique_x = unique_x.compute(X1, X2, Y)
scores_unique_y = unique_y.compute(X1, X2, Y)
scores_synergy = synergy.compute(X1, X2, Y)
```

### Technical Details

BROJA_2PID implements the Bertschinger-Rauh-Olbrich-Jost-Ay (BROJA) estimator for bivariate partial information decomposition. It decomposes the mutual information I(X1,X2;Y) into:

- **Shared Information**: Information about Y that both X1 and X2 provide
- **Unique Information**: Information that only X1 (or X2) provides about Y
- **Synergistic Information**: Information about Y that requires both X1 and X2

### Citation

If you use the PID metrics in your research, please cite the original BROJA paper:

```bibtex
@article{bertschinger2014quantifying,
  title={Quantifying unique information},
  author={Bertschinger, Nils and Rauh, Johannes and Olbrich, Eckehard and Jost, J{\"u}rgen and Ay, Nihat},
  journal={Entropy},
  volume={16},
  number={4},
  pages={2161--2183},
  year={2014},
  publisher={MDPI}
}
```

## Adding External Dependencies

When adding new external code:

1. **Create a subdirectory** with a clear name
2. **Include original license** and attribution
3. **Document why it's external** (not integrated)
4. **Provide integration examples**
5. **Add citation information** if applicable

## Guidelines

### When to Use External

Code should be in the external module when:

- It's a third-party implementation
- It has different licensing terms
- It requires minimal modification
- Academic attribution is important
- It's experimental or temporary

### When to Integrate

Code should be integrated into main modules when:

- It's heavily modified for our use case
- It follows our coding standards
- It's a core part of our framework
- We maintain it actively

## Future External Modules

Potential additions to the external module:

1. **Advanced Information Measures**: Other PID estimators, O-information
2. **Specialized Optimizers**: Research implementations
3. **Experimental Metrics**: Pre-publication algorithms
4. **Hardware-Specific Code**: CUDA kernels, TPU optimizations 