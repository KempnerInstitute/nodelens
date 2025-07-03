# Training Migration Example

This document shows how to migrate experiments from custom `_train_model()` implementations to use the unified `ExperimentTrainer`.

## Example: Migrating Layer-wise Pruning Experiment

### Before (Current Implementation)

```python
class LayerIsolatedPruningExperiment(BaseExperiment):
    def _train_model(self):
        """Train the model if configured."""
        if not self.config.train_before_dropout:
            logger.info("Skipping initial training (train_before_dropout=False)")
            return
            
        logger.info(f"Training model for {self.config.training_epochs} epochs")
        
        # Setup optimizer
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.config.training_epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Log epoch results
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            logger.info(f"Epoch {epoch+1}/{self.config.training_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            
            # Log metrics
            self.log_metrics(epoch, {
                "train_loss": avg_loss,
                "train_accuracy": accuracy
            })
```

### After (Using ExperimentTrainer)

```python
from alignment.experiments.training_utils import (
    create_experiment_trainer,
    train_with_metrics,
    convert_training_history
)

class LayerIsolatedPruningExperiment(BaseExperiment):
    def _train_model(self):
        """Train the model if configured."""
        if not self.config.train_before_dropout:
            logger.info("Skipping initial training (train_before_dropout=False)")
            return {}
        
        logger.info(f"Training model for {self.config.training_epochs} epochs")
        
        # Create trainer from config
        trainer = create_experiment_trainer(
            self.model, 
            self.config.__dict__,  # Convert dataclass to dict
            device=self.config.device
        )
        
        # Train with metrics
        history = train_with_metrics(
            trainer,
            self.data_loader,
            val_loader=None,  # No validation in original
            compute_accuracy=True
        )
        
        # Log final metrics (trainer already logs per-epoch)
        final_metrics = {
            "train_loss": history['train_loss'][-1],
            "train_accuracy": history['train_metrics'][-1].get('accuracy', 0.0)
        }
        self.log_metrics(len(history['train_loss']) - 1, final_metrics)
        
        # Return training results
        return convert_training_history(history)
```

## Benefits of Migration

1. **Less Code**: Reduced from ~50 lines to ~15 lines
2. **More Features**: Automatically get:
   - Learning rate scheduling
   - Early stopping
   - Gradient clipping
   - Better logging
   - Proper checkpointing
   
3. **Consistency**: All experiments use the same training logic
4. **Maintainability**: Bug fixes and improvements in one place
5. **Testing**: Unified trainer is well-tested

## Migration Checklist

- [ ] Replace `_train_model()` method with ExperimentTrainer usage
- [ ] Update config to include any missing training parameters
- [ ] Test that training behavior is preserved
- [ ] Remove any custom optimizer/scheduler creation
- [ ] Update any code that depends on training results format

## Advanced: Custom Training Behavior

If an experiment needs custom training behavior, you can:

1. **Use callbacks**: Pass custom callbacks to ExperimentTrainer
2. **Extend ExperimentTrainer**: Create a subclass for specific needs
3. **Use training hooks**: Override specific methods like `_train_epoch`

Example with callbacks:
```python
def custom_callback(trainer, epoch):
    """Custom logic after each epoch."""
    if epoch % 5 == 0:
        # Do something special every 5 epochs
        logger.info("Running custom logic...")

trainer = create_experiment_trainer(model, config)
trainer.callbacks.append(custom_callback)
```

## Next Steps

1. Start with simple experiments (single network, standard training)
2. Test thoroughly to ensure behavior is preserved
3. Move to more complex experiments
4. Update documentation and examples 