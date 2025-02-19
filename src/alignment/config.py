# --------------------------------------------
# config.py
# --------------------------------------------
from os import PathLike
from dataclasses import dataclass, field
from typing import (cast, List, Union, Type, TypeVar, Optional)
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException

C = TypeVar("C", bound="BaseConfig")

class BaseConfig:
    @classmethod
    def load(cls: Type[C], path: Union[str, PathLike]) -> C:
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            conf = om.merge(schema, raw)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise ValueError(str(e))

@dataclass
class ModelConfig(BaseConfig):
    name: str = "MLP"
    dropout: float = 0.0
    alignment_layers: Optional[dict] = None

@dataclass
class DatasetConfig(BaseConfig):
    name: str = "MNIST"
    path: Optional[str] = None

@dataclass
class TrainingConfig(BaseConfig):
    do_train: bool = True
    epochs: int = 10
    batch_size: int = 128
    replicates: int = 1
    name: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class AlignmentConfig(BaseConfig):
    do_alignment: bool = True
    methods: List[str] = field(default_factory=lambda: ["RQ"])
    measure_expected: bool = True
    frequency: int = 1
    bins: int = 50
    cnn_mode: str = "unfold"

@dataclass
class CheckpointingConfig(BaseConfig):
    use_prev: bool = False
    save_checkpoints: bool = False
    frequency: int = 1
    use_wandb: bool = False

@dataclass
class PlotsConfig(BaseConfig):
    show_loss: bool = True
    show_dropout: bool = True
    show_eigenfeatures: bool = True
    show_eig_dropout: bool = True

@dataclass
class ExtraConfig(BaseConfig):
    num_drops: int = 9
    dropout_by_layer: bool = False
    progressive_dropout_on_train: bool = False
    single_layer_mode: bool = False
    aggregate_alignment: bool = False

@dataclass
class ExperimentConfig(BaseConfig):
    experiment: Optional[str] = None
    results_path: Optional[str] = None
    no_save: bool = False
    just_plot: bool = False
    save_networks: bool = False
    show_params: bool = False
    show_all: bool = False
    device: Optional[str] = None
    use_timestamp: bool = False
    timestamp: Optional[str] = None
    ddp_world_size: int = 1
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    extra: ExtraConfig = field(default_factory=ExtraConfig)