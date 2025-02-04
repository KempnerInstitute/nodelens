import torch
from alignment_v2.utils import alignment as rq_alignment

class AlignmentMetrics:
    """
    Provides static methods for various alignment metrics.
    """

    @staticmethod
    def RQ(input_, weight_):
        # Reuse existing alignment(...) from utils, which does Rayleigh Quotient
        return rq_alignment(input_, weight_, method="alignment", relative=True)

    @staticmethod
    def MI_0(input_, weight_):
        # Placeholder for mutual info approach
        return torch.tensor(0.0)

    @staticmethod
    def MI_1(input_, weight_):
        return torch.tensor(0.0)

    @staticmethod
    def measure(input_, weight_, method="RQ"):
        if method == "RQ":
            return AlignmentMetrics.RQ(input_, weight_)
        elif method == "MI_0":
            return AlignmentMetrics.MI_0(input_, weight_)
        elif method == "MI_1":
            return AlignmentMetrics.MI_1(input_, weight_)
        else:
            raise ValueError(f"Unknown alignment method {method}")