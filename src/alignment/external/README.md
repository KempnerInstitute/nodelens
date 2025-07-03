# External Module

Third-party code and external dependencies integrated into the alignment framework.

## BROJA_2PID

Implementation of the BROJA estimator for Partial Information Decomposition (PID).

### Usage

Used internally by PID metrics:

```python
from alignment.metrics import get_metric

# These metrics use BROJA_2PID internally
shared_info = get_metric('pid_shared')
unique_x = get_metric('pid_unique_x')
unique_y = get_metric('pid_unique_y')
synergy = get_metric('pid_synergy')
```

### Citation

If using PID metrics, please cite:

```bibtex
@article{bertschinger2014quantifying,
  title={Quantifying unique information},
  author={Bertschinger, Nils and Rauh, Johannes and Olbrich, Eckehard and Jost, J{\"u}rgen and Ay, Nihat},
  journal={Entropy},
  volume={16},
  number={4},
  pages={2161--2183},
  year={2014}
}
```

## Guidelines

Code is placed in external when:
- It's a third-party implementation
- It has different licensing terms
- Academic attribution is important
- Minimal modification is required 