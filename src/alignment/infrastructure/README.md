# Infrastructure Module

System utilities for computing, storage, and configuration.

## Usage Status

| Component | Status | Description |
|-----------|--------|-------------|
| `storage/checkpoint.py` | ACTIVE | Model checkpoint save/load |
| `storage/logging.py` | ACTIVE | Logging setup and MetricLogger |
| `storage/job_directory.py` | ACTIVE | SLURM job directory management |
| `configuration/config.py` | AVAILABLE (warning) | Basic config utilities (use `alignment.configs` for main config) |
| `computing/distributed.py` | AVAILABLE | Multi-GPU distributed computing (not currently integrated) |
| `computing/optimized/gpu.py` | INTEGRATED | GPU-accelerated histogram/MI (enable via config) |
| `computing/optimized/jit.py` | INTEGRATED | JIT-compiled metrics (enable via config) |

## Components

### storage/ - Storage Infrastructure (ACTIVE)

**checkpoint.py** - Model checkpoint utilities
```python
from alignment.infrastructure import save_checkpoint, load_checkpoint

# Save model with optimizer state
save_checkpoint(model, optimizer, epoch=10, filepath="checkpoint.pt")

# Load checkpoint
checkpoint = load_checkpoint("checkpoint.pt", model=model, optimizer=optimizer)
```

**logging.py** - Logging utilities
```python
from alignment.infrastructure import setup_logging, get_logger, MetricLogger

# Setup logging
setup_logging(log_level="INFO", log_file="experiment.log")

# Get a logger
logger = get_logger(__name__)

# Track metrics over time
metric_logger = MetricLogger(log_dir="./logs", experiment_name="my_exp")
metric_logger.log({"loss": 0.5, "accuracy": 0.95}, step=100)
metric_logger.write_summary()
```

**job_directory.py** - SLURM job directory management
```python
from alignment.infrastructure.storage import create_job_directory, JobDirectory

# Create unique job directory (auto-detects SLURM_JOB_ID)
job_dir = create_job_directory(
    base_output_dir="/path/to/outputs",
    experiment_name="llama3_pruning"
)
# Creates: /path/to/outputs/llama3_pruning_20241209_143052_12345/
#          ├── results/
#          ├── logs/
#          ├── checkpoints/
#          ├── figures/
#          └── analysis/

# Or use context manager
with JobDirectory("/path/to/outputs", "my_experiment") as job:
    job.save_config(config)
    job.save_results(results)
```

### computing/ - Computing Infrastructure (AVAILABLE)

**distributed.py** - Distributed training utilities
```python
from alignment.infrastructure import (
    setup_distributed, cleanup_distributed,
    is_distributed, is_main_process,
    get_rank, get_world_size
)

# Setup distributed training
if setup_distributed(backend="nccl"):
    print(f"Rank {get_rank()} of {get_world_size()}")

# Check if main process (for logging)
if is_main_process():
    print("Only printed on rank 0")
```

**optimized/gpu.py** - GPU-accelerated operations
```python
from alignment.infrastructure.computing.optimized import (
    gpu_histogram1d, gpu_histogram2d,
    gpu_mutual_information, gpu_entropy,
    GPUAcceleratedMetrics
)

# Fast GPU histogram
hist, edges = gpu_histogram1d(data, bins=100)

# GPU mutual information
mi = gpu_mutual_information(x, y, bins=50)

# JIT-compiled covariance
cov = GPUAcceleratedMetrics.fast_covariance(X)
```

**optimized/jit.py** - JIT-compiled metrics
```python
from alignment.infrastructure.computing.optimized import (
    JITRayleighQuotient, JITMutualInformation, JITNodeCorrelation
)

# Create JIT-optimized metric
jit_rq = JITRayleighQuotient(epsilon=1e-8)
scores = jit_rq(inputs, weights)  # Faster than regular RQ
```

### configuration/ - Configuration Utilities (AVAILABLE, warning)

Basic configuration utilities. For the main experiment configuration system,
use `alignment.configs` instead.

```python
from alignment.infrastructure.configuration import load_config, save_config

# Load/save config files
config = load_config("config.yaml")
save_config(config, "output.yaml")
```

## Enabling Optimizations via Config

JIT and GPU acceleration are now integrated into the metric system. Enable them
via YAML config:

```yaml
metrics:
  optimization:
    use_jit: true                  # Enable JIT-compiled computations (20-50% faster)
    use_gpu_acceleration: true     # Enable GPU-accelerated functions
    force_cpu_for_large_ops: true  # Prevent OOM for large covariance matrices
    cpu_threshold: 100000000       # 1e8 elements threshold
```

Or programmatically:

```python
from alignment.metrics import get_optimization_status, get_metric_with_optimizations

# Check what's available
status = get_optimization_status()
print(f"JIT available: {status['jit_available']}")
print(f"GPU available: {status['gpu_available']}")

# Create a metric with optimizations
metric = get_metric_with_optimizations(
    "rayleigh_quotient",
    use_jit=True,
    use_gpu_acceleration=True,
    relative=True
)
```

## Future Integration Plans

The `computing/distributed.py` component is ready for integration when
multi-GPU metric computation becomes a priority.
