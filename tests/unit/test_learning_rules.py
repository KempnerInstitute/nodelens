import torch

from nodelens.learning_rules import (
    LearningRuleConfig,
    anti_decoupling_penalty,
    capacity_masses,
    channel_correlation,
    compact_hull_from_correlation,
    compact_hull_penalty,
    cross_layer_weights,
    peer_reconstructability_from_correlation,
    replaceability_regularization_loss,
    scheduled_regularizer_weight,
    synergy_pair_penalty,
    task_aware_redundancy_loss,
    task_relevance_from_logits,
    variance_floor_loss,
)


def test_channel_correlation_supports_conv_activations():
    activations = torch.randn(4, 6, 5, 5)

    corr = channel_correlation(activations)

    assert corr.shape == (6, 6)
    assert torch.all(torch.isfinite(corr))
    assert torch.allclose(torch.diagonal(corr), torch.ones(6), atol=1e-5)


def test_peer_reconstructability_detects_duplicate_channel():
    base = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    noise = torch.randn(32, 2)
    activations = torch.cat([base, base.clone(), noise], dim=1)

    corr = channel_correlation(activations)
    q = peer_reconstructability_from_correlation(corr)

    assert q[0] > 0.95
    assert q[1] > 0.95
    assert torch.all((q >= 0.0) & (q <= 1.0))


def test_task_aware_redundancy_loss_is_gate_sensitive():
    base = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    activations = torch.cat([base, base.clone(), torch.randn(64, 1)], dim=1)

    full_gate = torch.ones(3)
    muted_gate = torch.tensor([1.0, 0.0, 1.0])

    full_loss = task_aware_redundancy_loss(activations, full_gate)
    muted_loss = task_aware_redundancy_loss(activations, muted_gate)

    assert full_loss > muted_loss
    assert full_loss > 0


def test_task_aware_redundancy_loss_accepts_precomputed_correlation():
    activations = torch.randn(16, 4, 3, 3)
    gate = torch.rand(4)
    corr = channel_correlation(activations)

    direct = task_aware_redundancy_loss(activations, gate)
    cached = task_aware_redundancy_loss(activations, gate, correlation=corr)

    assert torch.allclose(direct, cached)


def test_variance_floor_penalizes_collapsed_channels():
    collapsed = torch.zeros(8, 4, 3, 3)
    active = torch.randn(8, 4, 3, 3)

    assert variance_floor_loss(collapsed) > variance_floor_loss(active)


def test_capacity_masses_split_task_relevance():
    task = torch.tensor([1.0, 2.0, 3.0])
    q = torch.tensor([0.0, 0.5, 1.0])

    duplicate, non_replaceable = capacity_masses(task, q)

    assert torch.isclose(duplicate, torch.tensor(4.0))
    assert torch.isclose(non_replaceable, torch.tensor(2.0))


def test_task_relevance_and_replaceability_loss_are_finite():
    activations = torch.randn(12, 5, 4, 4, requires_grad=True)
    logits = torch.randn(12, 3, requires_grad=True)
    targets = torch.randint(0, 3, (12,))

    relevance = task_relevance_from_logits(activations, logits, targets)
    loss, stats = replaceability_regularization_loss(
        {"layer": activations},
        logits,
        targets,
        LearningRuleConfig(method="bp_tard", weight=0.1),
    )

    assert relevance.shape == (5,)
    assert torch.all(torch.isfinite(relevance))
    assert torch.isfinite(loss)
    assert stats["regularized_layers"] == 1.0
    assert "duplicate_task_mass" in stats
    assert "non_replaceable_task_mass" in stats
    assert "rho_cap" in stats


def test_scheduled_regularizer_weight_warmup_and_ramp():
    cfg = LearningRuleConfig(method="bp_tard", weight=0.2, warmup_epochs=2, ramp_epochs=2)

    assert scheduled_regularizer_weight(cfg, 0) == 0.0
    assert scheduled_regularizer_weight(cfg, 1) == 0.0
    assert scheduled_regularizer_weight(cfg, 2) == 0.1
    assert scheduled_regularizer_weight(cfg, 3) == 0.2


def test_compact_hull_finds_small_hull_for_duplicates():
    base = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    activations = torch.cat([base, base.clone(), base.clone(), torch.randn(64, 5)], dim=1)
    corr = channel_correlation(activations)

    hull_size, full_r2, hull_score = compact_hull_from_correlation(corr, max_size=4, eps=0.05)

    assert hull_size.shape == (8,)
    assert full_r2.shape == (8,)
    assert hull_score.shape == (8,)
    assert torch.all(hull_size >= 0)
    assert torch.all(hull_size <= 4)
    # Duplicated channels should reach (1-eps) * full R^2 with a single peer.
    assert int(hull_size[0]) == 1
    assert int(hull_size[1]) == 1
    assert int(hull_size[2]) == 1
    # Independent noise channels should need more peers (or saturate the cap).
    assert int(hull_size[3:].max()) >= int(hull_size[:3].max())


def test_compact_hull_penalty_responds_to_gate_and_score():
    gate = torch.tensor([1.0, 1.0, 0.0])
    high_score = torch.tensor([0.8, 0.8, 0.8])
    low_score = torch.tensor([0.1, 0.1, 0.1])

    high_loss = compact_hull_penalty(gate, high_score)
    low_loss = compact_hull_penalty(gate, low_score)

    assert high_loss > low_loss
    # The off channel must not contribute.
    half_gate = torch.tensor([1.0, 0.0, 1.0])
    asymmetric = compact_hull_penalty(half_gate, torch.tensor([1.0, 1.0, 0.0]))
    assert torch.isclose(asymmetric, torch.tensor(1.0 / 3.0))


def test_cross_layer_weights_uniform_default():
    stats = [
        {"duplicate_task_mass": 1.0, "non_replaceable_task_mass": 0.5},
        {"duplicate_task_mass": 3.0, "non_replaceable_task_mass": 1.5},
        {"duplicate_task_mass": 2.0, "non_replaceable_task_mass": 1.0},
    ]
    assert cross_layer_weights(stats, mode="uniform") == [1.0, 1.0, 1.0]


def test_cross_layer_weights_dtm_share_concentrates_budget():
    stats = [
        {"duplicate_task_mass": 1.0, "non_replaceable_task_mass": 0.0},
        {"duplicate_task_mass": 3.0, "non_replaceable_task_mass": 0.0},
        {"duplicate_task_mass": 2.0, "non_replaceable_task_mass": 0.0},
    ]
    weights = cross_layer_weights(stats, mode="dtm_share")
    # Weights normalize so sum == n.
    assert abs(sum(weights) - 3.0) < 1e-6
    # Highest-DTM layer should get the largest weight.
    assert weights[1] > weights[2] > weights[0]


def test_cross_layer_weights_depth_increases_downstream():
    stats = [{"duplicate_task_mass": 1.0, "non_replaceable_task_mass": 0.0}] * 4
    weights = cross_layer_weights(stats, mode="depth", alpha=2.0)
    assert weights[-1] > weights[0]
    assert abs(sum(weights) - 4.0) < 1e-6


def test_cross_layer_weights_zero_signal_falls_back_to_uniform():
    stats = [
        {"duplicate_task_mass": 0.0, "non_replaceable_task_mass": 0.0},
        {"duplicate_task_mass": 0.0, "non_replaceable_task_mass": 0.0},
    ]
    assert cross_layer_weights(stats, mode="dtm_share") == [1.0, 1.0]


def test_replaceability_loss_handles_hull_method():
    activations = {"layer": torch.randn(12, 5, 4, 4, requires_grad=True)}
    logits = torch.randn(12, 3, requires_grad=True)
    targets = torch.randint(0, 3, (12,))
    cfg = LearningRuleConfig(method="bp_hull", weight=0.1, hull_max_size=3, hull_eps=0.05)

    loss, stats = replaceability_regularization_loss(activations, logits, targets, cfg)

    assert torch.isfinite(loss)
    assert stats["regularized_layers"] == 1.0
    assert "hull_size_mean" in stats
    assert "hull_score_mean" in stats


def test_grad_projection_zero_strength_is_noop():
    import torch.nn as nn

    from nodelens.learning_rules import LearningRuleConfig, _SigmaXEMA, project_signal_power_gradients

    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    cfg = LearningRuleConfig(method="none", weight=0.0, grad_projection_strength=0.0)
    sigma = _SigmaXEMA()
    inputs = {"conv": torch.randn(2, 3, 8, 8)}
    modules = {"conv": conv}
    out = conv(inputs["conv"])
    out.sum().backward()
    grad_before = conv.weight.grad.detach().clone()
    stats = project_signal_power_gradients(inputs, modules, sigma, cfg)
    assert stats == {"grad_projection_layers": 0.0}
    assert torch.equal(conv.weight.grad, grad_before)


def test_grad_projection_full_strength_modifies_gradient():
    import torch.nn as nn

    from nodelens.learning_rules import LearningRuleConfig, _SigmaXEMA, project_signal_power_gradients

    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    cfg = LearningRuleConfig(
        method="none",
        weight=0.0,
        grad_projection_strength=1.0,
        grad_projection_ridge=1e-3,
        grad_projection_max_patches=1024,
    )
    sigma = _SigmaXEMA()
    inputs = {"conv": torch.randn(4, 3, 8, 8)}
    modules = {"conv": conv}
    out = conv(inputs["conv"])
    target = torch.randn_like(out)
    (out - target).square().sum().backward()
    grad_before = conv.weight.grad.detach().clone()
    stats = project_signal_power_gradients(inputs, modules, sigma, cfg)
    assert stats["grad_projection_layers"] == 1.0
    assert "grad_projection_abs_cos_mean" in stats
    assert "grad_projection_norm_shrink_mean" in stats
    # Gradient should change unless the BP gradient is already orthogonal to
    # Sigma_X w_i, which is unlikely for an untrained Conv2d.
    assert not torch.allclose(conv.weight.grad, grad_before, atol=1e-7)
    # Norm shrink should be in (0, 1] (projection can only reduce norm).
    assert 0.0 < stats["grad_projection_norm_shrink_mean"] <= 1.0


def test_synergy_pair_penalty_negative_when_synergy_exists():
    torch.manual_seed(0)
    batch = 64
    # Construct: y1, y2 each carry only half of target, but together fully determine it.
    target = torch.randn(batch)
    y1 = 0.5 * target + 0.7 * torch.randn(batch)
    y2 = 0.5 * target + 0.7 * torch.randn(batch)
    pooled = torch.stack([y1, y2, torch.randn(batch)], dim=1)

    loss = synergy_pair_penalty(pooled, target, sample_pairs=None)

    # Synergy excess > 0 means loss = -synergy < 0.
    assert loss.item() < 0.0


def test_synergy_pair_penalty_handles_small_inputs():
    pooled = torch.zeros(1, 4, requires_grad=True)
    target = torch.zeros(1)
    loss = synergy_pair_penalty(pooled, target)
    assert torch.isfinite(loss)


def test_anti_decoupling_penalty_target_zero_is_minimized_by_decoupled():
    torch.manual_seed(0)
    batch = 64
    # Construct decoupled: I_X (variance) and I_T (corr with target) independent.
    target = torch.randn(batch)
    cols = []
    for i in range(8):
        scale = 0.1 + 0.5 * i
        if i % 2 == 0:
            cols.append(scale * torch.randn(batch))  # high variance, low task corr
        else:
            cols.append(scale * (0.5 * target + 0.5 * torch.randn(batch)))  # variable, with task signal
    pooled = torch.stack(cols, dim=1)

    loss_at_zero = anti_decoupling_penalty(pooled, target, target_rho=0.0)
    loss_at_target = anti_decoupling_penalty(pooled, target, target_rho=0.9)

    assert loss_at_zero.item() < loss_at_target.item()


def test_anti_decoupling_penalty_is_differentiable():
    torch.manual_seed(0)
    pooled = torch.randn(32, 6, requires_grad=True)
    target = torch.randn(32)
    loss = anti_decoupling_penalty(pooled, target, target_rho=0.3)
    loss.backward()
    assert pooled.grad is not None
    assert torch.all(torch.isfinite(pooled.grad))


def test_replaceability_loss_handles_synergy_method():
    activations = {"layer": torch.randn(16, 8, 4, 4, requires_grad=True)}
    logits = torch.randn(16, 5, requires_grad=True)
    targets = torch.randint(0, 5, (16,))
    cfg = LearningRuleConfig(method="bp_synergy", weight=0.1, synergy_sample_pairs=20)

    loss, stats = replaceability_regularization_loss(activations, logits, targets, cfg)

    assert torch.isfinite(loss)
    assert stats["regularized_layers"] == 1.0
    assert "synergy_raw_mean" in stats


def test_replaceability_loss_handles_antidecouple_method():
    activations = {"layer": torch.randn(16, 8, 4, 4, requires_grad=True)}
    logits = torch.randn(16, 5, requires_grad=True)
    targets = torch.randint(0, 5, (16,))
    cfg = LearningRuleConfig(method="bp_antidecouple", weight=0.1, anti_decouple_target_rho=0.4)

    loss, stats = replaceability_regularization_loss(activations, logits, targets, cfg)

    assert torch.isfinite(loss)
    assert stats["regularized_layers"] == 1.0
    assert "antidecouple_rho_l" in stats
    assert "antidecouple_gap" in stats


def test_replaceability_loss_records_cross_layer_weights():
    activations = {
        "a": torch.randn(8, 4, 3, 3, requires_grad=True),
        "b": torch.randn(8, 4, 3, 3, requires_grad=True),
        "c": torch.randn(8, 4, 3, 3, requires_grad=True),
    }
    logits = torch.randn(8, 3, requires_grad=True)
    targets = torch.randint(0, 3, (8,))
    cfg = LearningRuleConfig(method="bp_tard", weight=0.1, cross_layer_alloc="dtm_share")

    loss, stats = replaceability_regularization_loss(activations, logits, targets, cfg)

    assert torch.isfinite(loss)
    assert stats["cross_layer_alloc_mode"] == "dtm_share"
    assert "cross_layer_weight_mean" in stats
    assert abs(stats["cross_layer_weight_mean"] - 1.0) < 1e-5
