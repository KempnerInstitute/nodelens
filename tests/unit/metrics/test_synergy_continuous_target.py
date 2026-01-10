import torch

from alignment.metrics.information.synergy_continuous import SynergyContinuousTarget


def test_synergy_continuous_target_aggregates_unfolded_outputs(device):
    """
    When CNN preprocessing expands outputs to [B*P, C], synergy w.r.t. a per-example
    target must aggregate back to [B, C]. We verify that repeating each example's
    activations P times yields identical synergy scores after aggregation.
    """
    B, P, C, n_classes = 8, 7, 5, 10
    logits = torch.randn(B, n_classes, device=device)
    labels = torch.randint(0, n_classes, (B,), device=device)

    outputs_base = torch.randn(B, C, device=device)
    outputs_unfolded = outputs_base.repeat_interleave(P, dim=0)  # [B*P, C]

    metric = SynergyContinuousTarget(target_type="logit_margin", num_pairs=2, sampling_strategy="top_k")
    s_base = metric.compute(outputs=outputs_base, logits=logits, labels=labels)
    s_unfolded = metric.compute(outputs=outputs_unfolded, logits=logits, labels=labels)

    torch.testing.assert_close(s_base, s_unfolded, rtol=1e-5, atol=1e-6)

