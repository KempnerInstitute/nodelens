import sys
from alignment.experiments.alignment_stats import AlignmentStatistics
from alignment.experiments.alignment_comparison import AlignmentComparison
from alignment.experiments.adversarial_shaping import AdversarialShaping
from alignment.config import ExperimentConfig

EXPERIMENT_REGISTRY = {
    "alignment_stats": AlignmentStatistics,
    "alignment_comparison": AlignmentComparison,
    "adversarial_shaping": AdversarialShaping,
}

def get_experiment(cfg, build=False):
    """
    lookup model constructor from model registry by name

    if build=True, builds experiment and returns an experiment object using any kwargs
    otherwise just returns the class constructor
    """
    if cfg.experiment not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment ({cfg.experiment}) is not in EXPERIMENT_REGISTRY")
    experiment = EXPERIMENT_REGISTRY[cfg.experiment]
    if build:
        return experiment(cfg)
    return experiment

def create_experiment():
    """
    method to create experiment using initial argument parser

    the argument parser looks for a known argument called "--experiment", and the resulting
    string is used to retrieve an experiment constructor from the EXPERIMENT_REGISTRY

    any remaining arguments (args) are passed to the experiment constructor which has it's
    own argument parser in the class definition (but doesn't define the --experiment argument
    which is why the remaining args need to be passed to it directly)

    note:
    add_help=False so adding a --help argument will show a help message for the specific experiment
    that is requested rather than showing a help message for this little parser then blocking the
    rest of the execution. It means using --help requires a valid 'experiment' positional argument.
    """
    # TODO: adding the command line argument to the config
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise ValueError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = ExperimentConfig.load(yaml_path)
    return get_experiment(cfg, build=True)
