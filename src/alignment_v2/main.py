# main.py
# This loads the config, creates an AlignmentStats experiment, and runs or plots it.

import sys
import torch
from configs.alignment_config import ExperimentConfig
from experiments.alignment_stats import AlignmentStatistics

def main():
    cfg_path = sys.argv[1]
    cfg = ExperimentConfig.load(cfg_path)
    exp = AlignmentStatistics(cfg)
    if cfg.just_plot:
        exp.load_and_plot()
    else:
        results, nets = exp.run()
        if not cfg.no_save:
            exp.save_results(results)
        if not cfg.just_plot:
            exp.plot(results)

if __name__ == "__main__":
    main()