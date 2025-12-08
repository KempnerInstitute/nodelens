# Training Module

Training utilities, trainers, and evaluation functions.

## Components

### Trainers
- `BaseTrainer` - Base trainer class
- `ExperimentTrainer` - Trainer for alignment experiments
- `TensorizedNetworkWrapper` - Multi-network training

### Evaluation
- `evaluate_classification()` - Classification accuracy and loss
- `evaluate_perplexity()` - Language model perplexity
- `evaluate_regression()` - Regression MSE and MAE
- `evaluate_model()` - General dispatcher
- `EvaluationManager` - Track evaluation metrics over time

### Callbacks
- `AlignmentCallback` - Track alignment metrics during training

## Usage

### Training

```python
from alignment.training import ExperimentTrainer, ExperimentTrainingConfig

config = ExperimentTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    batch_size=128
)
trainer = ExperimentTrainer(model, config)
trainer.train(train_loader, val_loader)
```

### Evaluation

```python
from alignment.training import evaluate_classification, evaluate_perplexity

# Classification
results = evaluate_classification(model, test_loader, device="cuda")
# Returns: {"loss": 0.32, "accuracy": 91.5}

# Language modeling
results = evaluate_perplexity(model, text_loader, device="cuda")
# Returns: {"perplexity": 12.4, "loss": 2.52}
```

### Evaluation Manager

```python
from alignment.training import EvaluationManager

manager = EvaluationManager(task="classification")

for epoch in range(epochs):
    train(...)
    results = manager.evaluate(model, val_loader, step=epoch)
    
best = manager.get_best(metric="accuracy")
history = manager.get_history()
```
