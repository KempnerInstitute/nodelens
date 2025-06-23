"""
Reinforcement learning-specific alignment metrics.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from ...core.registry import register_metric
from ...core.base import BaseMetric


@register_metric("reinforcement_learning_alignment")
class ReinforcementLearningAlignment(BaseMetric):
    """
    Measures alignment for reinforcement learning tasks.
    
    This metric evaluates how well neuron activations align with value functions,
    policy gradients, and reward signals in RL settings.
    """
    
    name = "reinforcement_learning_alignment"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False
    
    def __init__(
        self,
        alignment_type: str = "value_correlation",
        discount_factor: float = 0.99,
        n_actions: int = 4,
        use_advantage: bool = True
    ):
        """
        Initialize RL alignment metric.
        
        Args:
            alignment_type: Type of alignment measure
                - 'value_correlation': Correlation with value function
                - 'policy_gradient': Alignment with policy gradients
                - 'reward_prediction': Reward prediction accuracy
                - 'temporal_difference': TD error alignment
            discount_factor: Discount factor for future rewards
            n_actions: Number of possible actions
            use_advantage: Whether to use advantage function
        """
        super().__init__()
        self.alignment_type = alignment_type
        self.discount_factor = discount_factor
        self.n_actions = n_actions
        self.use_advantage = use_advantage
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute RL alignment scores.
        
        Args:
            inputs: Input activations (state representations) [batch_size, state_dim]
            weights: Weight matrix [output_dim, input_dim]
            outputs: Output activations [batch_size, output_dim]
            states: Environment states
            actions: Actions taken [batch_size]
            rewards: Rewards received [batch_size]
            next_states: Next states
            values: Value function estimates [batch_size]
            
        Returns:
            Alignment scores for each neuron [output_dim]
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        n_neurons = weights.shape[0]
        alignment_scores = torch.zeros(n_neurons, device=outputs.device)
        
        # Generate synthetic RL data if not provided
        batch_size = outputs.shape[0]
        if states is None:
            states = inputs
        if actions is None:
            actions = torch.randint(0, self.n_actions, (batch_size,), device=outputs.device)
        if rewards is None:
            rewards = torch.randn(batch_size, device=outputs.device)
        if next_states is None:
            next_states = states + torch.randn_like(states) * 0.1
        
        if self.alignment_type == "value_correlation":
            # Measure correlation with value function
            if values is None:
                # Estimate values using simple linear projection
                value_projection = torch.randn(outputs.shape[1], 1, device=outputs.device)
                values = (outputs @ value_projection).squeeze()
            
            # For each neuron, compute correlation with value estimates
            for i in range(n_neurons):
                neuron_activations = outputs[:, i]
                
                if neuron_activations.std() > 0 and values.std() > 0:
                    correlation = torch.corrcoef(
                        torch.stack([neuron_activations, values])
                    )[0, 1]
                    alignment_scores[i] = correlation.abs()
        
        elif self.alignment_type == "policy_gradient":
            # Measure alignment with policy gradients
            # Create policy network output (action probabilities)
            policy_projection = torch.randn(n_neurons, self.n_actions, device=outputs.device)
            action_logits = outputs @ policy_projection
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Compute log probabilities of taken actions
            action_log_probs = F.log_softmax(action_logits, dim=-1)
            taken_action_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Compute advantages if using advantage function
            if self.use_advantage and values is not None:
                # Compute returns (simple 1-step return for demonstration)
                returns = rewards + self.discount_factor * values.roll(-1)
                advantages = returns - values
            else:
                advantages = rewards
            
            # Policy gradient: grad log π(a|s) * A
            # We measure how neuron activation correlates with this quantity
            policy_gradient_signal = taken_action_log_probs * advantages.detach()
            
            for i in range(n_neurons):
                neuron_activations = outputs[:, i]
                
                if neuron_activations.std() > 0:
                    # Compute correlation with policy gradient signal
                    correlation = torch.corrcoef(
                        torch.stack([neuron_activations, policy_gradient_signal])
                    )[0, 1]
                    alignment_scores[i] = correlation.abs()
        
        elif self.alignment_type == "reward_prediction":
            # Measure how well neurons predict future rewards
            for i in range(n_neurons):
                neuron_activations = outputs[:, i]
                
                # Simple linear regression to predict rewards
                if neuron_activations.std() > 0:
                    # Normalize
                    X = (neuron_activations - neuron_activations.mean()) / neuron_activations.std()
                    y = rewards
                    
                    # Compute correlation as measure of predictive power
                    if y.std() > 0:
                        correlation = torch.corrcoef(torch.stack([X, y]))[0, 1]
                        alignment_scores[i] = correlation.abs()
        
        elif self.alignment_type == "temporal_difference":
            # Measure alignment with TD errors
            if values is None:
                # Estimate values
                value_projection = torch.randn(outputs.shape[1], 1, device=outputs.device)
                values = (outputs @ value_projection).squeeze()
                
                # Compute next values
                next_outputs = next_states @ weights.T
                next_values = (next_outputs @ value_projection).squeeze()
            else:
                # Compute next values from next states
                next_outputs = next_states @ weights.T
                value_projection = torch.randn(outputs.shape[1], 1, device=outputs.device)
                next_values = (next_outputs @ value_projection).squeeze()
            
            # TD error: r + γV(s') - V(s)
            td_errors = rewards + self.discount_factor * next_values - values
            
            # Measure how neuron activations correlate with TD errors
            for i in range(n_neurons):
                neuron_activations = outputs[:, i]
                
                if neuron_activations.std() > 0 and td_errors.std() > 0:
                    correlation = torch.corrcoef(
                        torch.stack([neuron_activations, td_errors.abs()])
                    )[0, 1]
                    alignment_scores[i] = correlation.abs()
        
        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")
        
        return alignment_scores 