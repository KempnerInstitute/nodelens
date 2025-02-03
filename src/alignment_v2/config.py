from os import PathLike
from dataclasses import dataclass, field
from typing import (cast,
                    List,
                    Union,
                    Type,
                    TypeVar,
                    Optional,
)

from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException

C = TypeVar("C", bound="BaseConfig")

class BaseConfig:
    @classmethod
    def load(
        cls: Type[C],
        path: Union[str, PathLike],
        ) -> C:
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
    dropout: float = 0
    alignment_layers: Optional[dict] = None
    """
    A dictionary specifying the layers participating in alignment computation and their corresponding input layers
    none indicates all layers to participate. Both key and values should be layer names from model.named_modules() 
    """

@dataclass
class DatasetConfig(BaseConfig):
    name: str = "MNIST"
    path: Optional[str] = None
    #download: bool = False

@dataclass
class OptimizerConfig(BaseConfig):
    name: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0

@dataclass
class TrainingConfig(BaseConfig):
    batch_size: int = 1024
    epochs: int = 100
    replicates: int = 5

@dataclass
class AlignmentConfig(BaseConfig):
    compute_alignment: bool = True
    measure_weight_deltas: bool = False
    alignment_eval_frequency: int = 1
    methods: List[str] = field(default_factory=lambda: ["RQ"])
    compute_during_training: bool = True
    compute_during_inference: bool = True

@dataclass
class CheckpointingConfig(BaseConfig):
    use_prev: bool = False
    save_checkpoints: bool = False
    frequency: int = 1
    use_wandb: bool = False

@dataclass
class ExtraConfig(BaseConfig):
    num_drops: int = 9
    dropout_by_layer: bool = False
    #############################  For alignment_comparison and loading_prediction experiments
    comparison: str = "lr"
    regularizers: List[str] = field(default_factory=lambda: ["none", "dropout", "weight_decay"])
    lrs: List[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    compare_dropout: float = 0.5
    compare_wd: float = 1e-5
    #############################  For adversarial_shaping experiment
    cutoffs: List[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4, 0.0])
    manual_frequency: int = 5

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
   dataset: DatasetConfig = field(default_factory=DatasetConfig)
   model: ModelConfig = field(default_factory=ModelConfig)
   optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)   
   training: TrainingConfig = field(default_factory=TrainingConfig)
   alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
   checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
   extra: ExtraConfig = field(default_factory=ExtraConfig)
