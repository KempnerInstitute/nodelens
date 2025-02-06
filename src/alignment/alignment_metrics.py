# --------------------------------------------
# alignment_metrics.py
# --------------------------------------------

import torch
from alignment.utils import alignment as rq_alignment

class AlignmentMetrics:
    """
    Provides static methods for various alignment metrics.
    """

    @staticmethod
    def RQ(input_, weight_):
        """
        This is the Rayleigh Quotient alignment measure measures for how much
        variance of the input is explained by each node's weight vector.
        """
        return rq_alignment(input_, weight_, method="alignment", relative=True)

    @staticmethod
    def MI_0(input_, weight_):
        """
        Placeholder for mutual information approach - version 0
        """
        return rq_alignment(input_, weight_, method="alignment", relative=True)

    @staticmethod
    def MI_1(input_, weight_):
        """
        Placeholder for mutual information approach - version 1
        """
        return torch.tensor(0.0)

    @staticmethod
    def measure(input_, weight_, method="RQ"):
        """
        Dispatch method to pick one of the alignment/MI metrics.
        """
        if method == "RQ":
            return AlignmentMetrics.RQ(input_, weight_)
        elif method == "MI_0":
            return AlignmentMetrics.MI_0(input_, weight_)
        elif method == "MI_1":
            return AlignmentMetrics.MI_1(input_, weight_)
        else:
            raise ValueError(f"Unknown alignment method {method}")


