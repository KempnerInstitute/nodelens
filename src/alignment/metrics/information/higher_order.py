"""
Higher-order information decomposition metrics.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
from ...core.registry import register_metric
from ...core.metrics import BaseMetric
from .mutual_information import estimate_mutual_information_binning


@register_metric("total_correlation")
class TotalCorrelation(BaseMetric):
    """
    Compute total correlation (multi-information) among a set of neurons.
    
    Total correlation measures the amount of information shared among all neurons,
    beyond pairwise dependencies.
    """
    
    name = "total_correlation"
    
    def __init__(self, method: str = "gaussian", bins: int = 10):
        """
        Initialize total correlation metric.
        
        Args:
            method: Estimation method ('gaussian' or 'binning')
            bins: Number of bins for discretization (if using binning)
        """
        super().__init__()
        self.method = method
        self.bins = bins
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute total correlation for groups of neurons.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations
            
        Returns:
            Total correlation scores
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        batch_size, n_neurons = outputs.shape
        
        if self.method == "gaussian":
            # Assume Gaussian distribution
            # Center the outputs
            outputs_centered = outputs - outputs.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov_matrix = (outputs_centered.T @ outputs_centered) / (batch_size - 1)
            
            # Add small diagonal for stability
            cov_matrix = cov_matrix + 1e-8 * torch.eye(n_neurons, device=outputs.device)
            
            # Compute determinant of full covariance matrix
            det_full = torch.linalg.det(cov_matrix)
            
            # Compute product of individual variances
            variances = torch.diag(cov_matrix)
            log_prod_vars = torch.log(variances).sum()
            
            # Total correlation = 0.5 * log(prod(variances)) - 0.5 * log(det(cov))
            tc = 0.5 * (log_prod_vars - torch.log(det_full))
            
            # Return as per-neuron score (distributed evenly)
            return torch.full((n_neurons,), tc.item() / n_neurons, device=outputs.device)
        
        else:  # binning method
            # Discretize outputs
            outputs_np = outputs.cpu().numpy()
            
            # Compute individual entropies
            individual_entropies = []
            for i in range(n_neurons):
                hist, _ = np.histogram(outputs_np[:, i], bins=self.bins)
                hist = hist + 1e-8  # Add small constant
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log(hist))
                individual_entropies.append(entropy)
            
            # Compute joint entropy
            # For computational efficiency, we'll use a subset of neurons
            max_neurons_joint = min(5, n_neurons)  # Limit to 5 neurons for joint entropy
            
            if n_neurons > max_neurons_joint:
                # Sample neurons randomly
                idx = torch.randperm(n_neurons)[:max_neurons_joint]
                outputs_subset = outputs[:, idx]
            else:
                outputs_subset = outputs
            
            # Compute joint histogram
            outputs_subset_np = outputs_subset.cpu().numpy()
            hist_joint, _ = np.histogramdd(outputs_subset_np, bins=self.bins)
            hist_joint = hist_joint.flatten() + 1e-8
            hist_joint = hist_joint / hist_joint.sum()
            joint_entropy = -np.sum(hist_joint * np.log(hist_joint))
            
            # Total correlation approximation
            sum_individual = sum(individual_entropies[:outputs_subset.shape[1]])
            tc = sum_individual - joint_entropy
            
            # Return as tensor
            return torch.full((n_neurons,), tc / n_neurons, device=outputs.device)


@register_metric("interaction_information")
class InteractionInformation(BaseMetric):
    """
    Compute interaction information (co-information) among triplets of variables.
    
    This measures the amount of information that is present only when all three
    variables are considered together.
    """
    
    name = "interaction_information"
    
    def __init__(self, n_samples: int = 100, bins: int = 10):
        """
        Initialize interaction information metric.
        
        Args:
            n_samples: Number of triplet samples to evaluate
            bins: Number of bins for discretization
        """
        super().__init__()
        self.n_samples = n_samples
        self.bins = bins
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute interaction information scores.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations
            
        Returns:
            Interaction information scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        n_neurons = outputs.shape[1]
        interaction_scores = torch.zeros(n_neurons, device=outputs.device)
        
        # Sample triplets of neurons
        n_triplets = min(self.n_samples, n_neurons * (n_neurons - 1) * (n_neurons - 2) // 6)
        
        for _ in range(n_triplets):
            # Randomly select 3 different neurons
            idx = torch.randperm(n_neurons)[:3]
            i, j, k = idx[0], idx[1], idx[2]
            
            # Get activations for these neurons
            X = outputs[:, i]
            Y = outputs[:, j]
            Z = outputs[:, k]
            
            # Compute pairwise mutual information
            MI_XY = estimate_mutual_information_binning(
                X.unsqueeze(1), Y.unsqueeze(1), bins=self.bins
            )
            MI_XZ = estimate_mutual_information_binning(
                X.unsqueeze(1), Z.unsqueeze(1), bins=self.bins
            )
            MI_YZ = estimate_mutual_information_binning(
                Y.unsqueeze(1), Z.unsqueeze(1), bins=self.bins
            )
            
            # Compute conditional mutual information I(X;Y|Z)
            # Using approximation: I(X;Y|Z) ≈ I(X;Y) - I(X;Y;Z)
            # Where I(X;Y;Z) is the interaction information
            
            # For simplicity, we'll use the difference of mutual informations
            # as an approximation of interaction information
            interaction = MI_XY + MI_XZ + MI_YZ
            
            # Distribute score to participating neurons
            interaction_scores[i] += interaction / 3
            interaction_scores[j] += interaction / 3
            interaction_scores[k] += interaction / 3
        
        # Normalize by number of samples
        interaction_scores = interaction_scores / n_triplets
        
        return interaction_scores


@register_metric("connected_information")  
class ConnectedInformation(BaseMetric):
    """
    Compute connected information (Amari, 2001) which decomposes mutual information
    into hierarchical terms.
    """
    
    name = "connected_information"
    
    def __init__(self, max_order: int = 3, method: str = "gaussian"):
        """
        Initialize connected information metric.
        
        Args:
            max_order: Maximum order of interactions to consider
            method: Estimation method ('gaussian' or 'binning')
        """
        super().__init__()
        self.max_order = max_order
        self.method = method
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute connected information scores.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations
            
        Returns:
            Connected information scores
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        n_neurons = outputs.shape[1]
        
        if self.method == "gaussian":
            # For Gaussian case, use cumulants
            # Center the outputs
            outputs_centered = outputs - outputs.mean(dim=0, keepdim=True)
            
            # Compute covariance (2nd order cumulant)
            cov_matrix = (outputs_centered.T @ outputs_centered) / (outputs.shape[0] - 1)
            
            # For higher orders, we'd need to compute higher cumulants
            # For now, we'll use the Frobenius norm of covariance as a proxy
            connected_info = cov_matrix.norm(p='fro')
            
            # Distribute score across neurons
            scores = torch.full((n_neurons,), connected_info.item() / n_neurons, 
                              device=outputs.device)
        
        else:
            # For non-Gaussian, this is computationally intensive
            # We'll use a simplified approximation based on total correlation
            tc_metric = TotalCorrelation(method="binning")
            scores = tc_metric.compute(inputs, weights, outputs)
        
        return scores


@register_metric("synergistic_information")
class SynergisticInformation(BaseMetric):
    """
    Compute synergistic information - information that can only be obtained
    from the joint state of multiple neurons.
    """
    
    name = "synergistic_information"
    
    def __init__(self, group_size: int = 3, n_groups: int = 50):
        """
        Initialize synergistic information metric.
        
        Args:
            group_size: Size of neuron groups to analyze
            n_groups: Number of random groups to sample
        """
        super().__init__()
        self.group_size = group_size
        self.n_groups = n_groups
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute synergistic information scores.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations
            
        Returns:
            Synergistic information scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        batch_size, n_neurons = outputs.shape
        synergy_scores = torch.zeros(n_neurons, device=outputs.device)
        
        # Sample groups of neurons
        n_groups_actual = min(self.n_groups, n_neurons // self.group_size)
        
        for _ in range(n_groups_actual):
            # Select a random group
            idx = torch.randperm(n_neurons)[:self.group_size]
            group_outputs = outputs[:, idx]
            
            # Compute joint entropy of the group
            # Using Gaussian assumption for efficiency
            group_centered = group_outputs - group_outputs.mean(dim=0, keepdim=True)
            cov_group = (group_centered.T @ group_centered) / (batch_size - 1)
            cov_group = cov_group + 1e-8 * torch.eye(self.group_size, device=outputs.device)
            
            # Joint entropy under Gaussian assumption
            # H = 0.5 * log(det(2πe * Σ))
            det_cov = torch.linalg.det(cov_group)
            joint_entropy = 0.5 * torch.log(2 * np.pi * np.e * det_cov)
            
            # Compute sum of individual entropies
            individual_entropies = 0
            for i in range(self.group_size):
                var = cov_group[i, i]
                individual_entropies += 0.5 * torch.log(2 * np.pi * np.e * var)
            
            # Synergy approximation: joint entropy - sum of individual entropies
            # (negative of this gives redundancy, positive gives synergy)
            synergy = joint_entropy - individual_entropies
            
            # Distribute score to participating neurons
            for i in idx:
                synergy_scores[i] += synergy / self.group_size
        
        # Normalize by number of groups
        synergy_scores = synergy_scores / n_groups_actual
        
        return synergy_scores 