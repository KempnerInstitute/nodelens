"""
Comprehensive example demonstrating all features of the refactored alignment framework.

This example shows:
1. All experiment types
2. All metrics including PID
3. Training options
4. Configuration features
"""

from alignment_refactor import (
    # Models and data
    ModelWrapper, DatasetWrapper,
    
    # Experiments
    ProgressiveDropoutExperiment,
    LayerIsolatedPruningExperiment, LayerIsolatedConfig,
    CascadingLayerPruningExperiment, CascadingConfig,
    EigenvectorDropoutExperiment, EigenvectorConfig,
    ExperimentRunner,
    
    # Metrics
    discover_metrics,
    
    # Training
    train_networks_fully_tensorized,
    
    # Configuration
    ExperimentConfig
)
import torch


def example_all_metrics():
    """Example showing all available metrics."""
    print("Available metrics:")
    metrics = discover_metrics()
    
    # Group by category
    rayleigh_metrics = [m for m in metrics if 'rayleigh' in m or 'rq' in m]
    info_metrics = [m for m in metrics if 'mi_' in m or 'redundancy' in m or 'conditional' in m]
    pid_metrics = [m for m in metrics if 'pid' in m]
    similarity_metrics = [m for m in metrics if 'similarity' in m or 'alignment' in m]
    
    print("\nRayleigh Quotient Metrics:")
    for m in rayleigh_metrics:
        print(f"  - {m}")
    
    print("\nInformation Metrics:")
    for m in info_metrics:
        print(f"  - {m}")
    
    print("\nPID Metrics:")
    for m in pid_metrics:
        print(f"  - {m}")
    
    print("\nSimilarity Metrics:")
    for m in similarity_metrics:
        print(f"  - {m}")


def example_progressive_dropout():
    """Example of progressive dropout experiment."""
    print("\n=== Progressive Dropout Experiment ===")
    
    # Configuration with all options
    config = ExperimentConfig(
        name="progressive_dropout_demo",
        model_name="resnet18",
        dataset_name="cifar10",
        metrics=["rayleigh_quotient", "mi_gaussian", "pid_shared"],
        
        # Training options
        train_before_dropout=True,
        training_epochs=5,
        learning_rate=0.001,
        optimizer="adam",
        
        # Metric options
        scale_by_norm=True,
        force_cpu_for_large_metric_ops=True,
        cnn_rq_aggregation_op="mean",
        
        # Experiment-specific
        dropout_rates=[0.0, 0.2, 0.4, 0.6, 0.8],
        dropout_mode="scaled",
        pruning_mode="global_joint",
        pruning_strategy="low",
        exclude_classification_layer=True
    )
    
    # Create and run experiment
    experiment = ProgressiveDropoutExperiment(config)
    # results = experiment.run()  # Uncomment to actually run
    
    print("Progressive dropout configured with:")
    print(f"  - Dropout rates: {config.dropout_rates}")
    print(f"  - Metrics: {config.metrics}")
    print(f"  - Scale by norm: {config.scale_by_norm}")


def example_layer_isolated():
    """Example of layer-isolated pruning."""
    print("\n=== Layer-Isolated Pruning Experiment ===")
    
    config = LayerIsolatedConfig(
        name="layer_isolated_demo",
        model_name="resnet18",
        dataset_name="cifar10",
        metrics=["rayleigh_quotient", "weight_cosine_similarity"],
        
        # Pruning configuration
        dropout_rates=[0.0, 0.3, 0.6, 0.9],
        pruning_metric="rayleigh_quotient",
        pruning_strategy="low",
        exclude_classification_layer=True,
        
        # Training
        train_before_dropout=True,
        training_epochs=10,
        eval_batches=50
    )
    
    experiment = LayerIsolatedPruningExperiment(config)
    # results = experiment.run()
    
    print("Layer-isolated pruning configured")
    print("Each layer pruned independently based on its own scores")


def example_cascading():
    """Example of cascading layer pruning."""
    print("\n=== Cascading Layer Pruning Experiment ===")
    
    config = CascadingConfig(
        name="cascading_demo",
        model_name="resnet18",
        dataset_name="cifar10",
        metrics=["rayleigh_quotient", "node_redundancy"],
        
        # Cascading options
        cascade_direction="forward",
        recompute_scores=True,
        
        # Pruning
        dropout_rates=[0.0, 0.25, 0.5, 0.75],
        pruning_metric="rayleigh_quotient"
    )
    
    experiment = CascadingLayerPruningExperiment(config)
    # results = experiment.run()
    
    print("Cascading pruning configured")
    print(f"  - Direction: {config.cascade_direction}")
    print(f"  - Recompute scores: {config.recompute_scores}")


def example_eigenvector():
    """Example of eigenvector dropout."""
    print("\n=== Eigenvector Dropout Experiment ===")
    
    config = EigenvectorConfig(
        name="eigenvector_demo",
        model_name="resnet18",
        dataset_name="cifar10",
        metrics=["rayleigh_quotient", "pid_synergy"],
        
        # Eigenvector options
        compute_layer_pca=True,
        n_components_ratio=0.95,
        eigenvector_strategy="low",
        
        # Dropout rates
        dropout_rates=[0.0, 0.2, 0.4, 0.6, 0.8]
    )
    
    experiment = EigenvectorDropoutExperiment(config)
    # results = experiment.run()
    
    print("Eigenvector dropout configured")
    print(f"  - Components to keep: {config.n_components_ratio * 100}% variance")
    print(f"  - Strategy: drop {config.eigenvector_strategy} eigenvalue components")


def example_tensorized_training():
    """Example of fully tensorized training."""
    print("\n=== Fully Tensorized Training ===")
    
    # Create multiple networks with same architecture
    num_networks = 4
    networks = []
    
    for i in range(num_networks):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model.fc = torch.nn.Linear(512, 10)  # CIFAR-10 has 10 classes
        networks.append(model)
    
    print(f"Created {num_networks} networks for tensorized training")
    
    # Setup data
    dataset = DatasetWrapper.from_name("cifar10")
    train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )
    
    # Train (commented out for demo)
    # trained_networks, history = train_networks_fully_tensorized(
    #     networks=networks,
    #     train_loader=train_loader,
    #     epochs=10,
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    
    print("Tensorized training allows efficient training of multiple networks")
    print("All networks trained simultaneously with batched operations")


def example_experiment_runner():
    """Example of running multiple experiments."""
    print("\n=== Experiment Runner ===")
    
    # Define multiple experiment configs
    configs = [
        ExperimentConfig(
            name="exp1_progressive",
            model_name="resnet18",
            dataset_name="cifar10",
            metrics=["rayleigh_quotient"],
            dropout_rates=[0.0, 0.5]
        ),
        LayerIsolatedConfig(
            name="exp2_isolated",
            model_name="resnet18",
            dataset_name="cifar10",
            metrics=["rayleigh_quotient"],
            dropout_rates=[0.0, 0.5]
        ),
        CascadingConfig(
            name="exp3_cascading",
            model_name="resnet18",
            dataset_name="cifar10",
            metrics=["rayleigh_quotient"],
            dropout_rates=[0.0, 0.5]
        )
    ]
    
    # Create runner
    runner = ExperimentRunner(
        configs=configs,
        results_dir="./experiment_results",
        parallel=False,  # Set True for parallel execution
        max_workers=2
    )
    
    # Run all experiments (commented out for demo)
    # results = runner.run()
    
    print(f"Runner configured to run {len(configs)} experiments")
    print("Experiments can run sequentially or in parallel")


def example_comprehensive_config():
    """Example showing all configuration options."""
    print("\n=== Comprehensive Configuration ===")
    
    config = ExperimentConfig(
        # Basic info
        name="comprehensive_experiment",
        description="Experiment with all configuration options",
        tags=["demo", "full_features"],
        
        # Model
        model_name="resnet18",
        model_config={"pretrained": True},
        
        # Dataset
        dataset_name="cifar10",
        data_path="./data",
        batch_size=128,
        num_workers=4,
        
        # Training
        train_before_dropout=True,
        training_epochs=10,
        learning_rate=0.001,
        optimizer="adam",
        
        # Metrics
        metrics=["rayleigh_quotient", "mi_gaussian", "pid_shared", 
                "weight_cosine_similarity", "node_redundancy"],
        metric_configs={
            "rayleigh_quotient": {"scale_by_norm": True},
            "mi_gaussian": {"normalize": True}
        },
        
        # Metric computation options
        scale_by_norm=True,
        force_cpu_for_large_metric_ops=True,
        cnn_rq_aggregation_op="mean",
        
        # Dropout experiment options
        dropout_rates=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        dropout_mode="scaled",
        pruning_mode="global_joint",
        pruning_strategy="low",
        exclude_classification_layer=True,
        
        # Checkpointing
        checkpoint_dir="./checkpoints",
        checkpoint_interval=1000,
        save_best=True,
        
        # Logging
        log_dir="./logs",
        log_interval=100,
        wandb_project="alignment_experiments",
        
        # Distributed
        distributed=False,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42
    )
    
    print("Configuration includes:")
    print(f"  - {len(config.metrics)} metrics")
    print(f"  - {len(config.dropout_rates)} dropout rates") 
    print(f"  - Training: {config.training_epochs} epochs")
    print(f"  - All metric computation options enabled")


if __name__ == "__main__":
    print("=== Alignment Framework Feature Demo ===\n")
    
    # Show all available metrics
    example_all_metrics()
    
    # Demonstrate each experiment type
    example_progressive_dropout()
    example_layer_isolated()
    example_cascading()
    example_eigenvector()
    
    # Show training options
    example_tensorized_training()
    
    # Show experiment runner
    example_experiment_runner()
    
    # Show comprehensive configuration
    example_comprehensive_config()
    
    print("\n=== All Features Demonstrated ===")
    print("The refactored framework includes:")
    print("  - All original metrics + PID metrics")
    print("  - All experiment types")
    print("  - Fully tensorized training")
    print("  - Enhanced configuration options")
    print("  - Clean, modular architecture") 