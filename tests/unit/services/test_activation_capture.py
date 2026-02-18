import torch
import torch.nn as nn

from alignment.models import ModelWrapper
from alignment.services.activation_capture import ActivationCaptureService


def test_activation_capture_conv2d_unfold_matches_conv(device):
    """
    For Conv2d with bias=False, verify that:
    - inputs are unfolded into patches [B*P, C_in*kH*kW]
    - outputs are spatial-flattened [B*P, C_out]
    - outputs == inputs @ W^T (exact conv equivalence via unfold)
    """

    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = SimpleConv().to(device)
    wrapper = ModelWrapper(model, tracked_layers=["conv"], preprocessing_mode="auto")
    service = ActivationCaptureService(wrapper, default_mode="unfold")

    B, C, H, W = 2, 3, 8, 8
    x = torch.randn(B, C, H, W, device=device)

    data = service.capture(x, layers=["conv"], include_weights=True, preprocess=True)

    # Shapes
    assert "conv" in data.inputs
    assert "conv" in data.outputs
    assert "conv" in data.weights

    # For stride=1,pad=1,k=3 => H_out=W_out=H=W, P=H*W
    P = H * W
    F = C * 3 * 3
    assert tuple(data.inputs["conv"].shape) == (B * P, F)
    assert tuple(data.outputs["conv"].shape) == (B * P, 4)
    assert tuple(data.weights["conv"].shape) == (4, F)

    # Numerical equivalence: conv(x) == unfold(x) @ W^T
    pred = data.inputs["conv"] @ data.weights["conv"].T
    torch.testing.assert_close(pred, data.outputs["conv"], rtol=1e-4, atol=1e-5)


def test_activation_capture_conv2d_patchwise_matches_conv(device):
    """
    Patchwise mode keeps patches separate:
      inputs:  [B, F, P]
      outputs: [B, C_out, P]
    and should still satisfy patchwise linear equivalence for bias=False.
    """

    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = SimpleConv().to(device)
    wrapper = ModelWrapper(model, tracked_layers=["conv"], preprocessing_mode="auto")
    service = ActivationCaptureService(wrapper, default_mode="patchwise")

    B, C, H, W = 2, 3, 6, 7
    x = torch.randn(B, C, H, W, device=device)

    data = service.capture(x, layers=["conv"], include_weights=True, preprocess=True)

    P = H * W
    F = C * 3 * 3
    assert tuple(data.inputs["conv"].shape) == (B, F, P)
    assert tuple(data.outputs["conv"].shape) == (B, 5, P)
    assert tuple(data.weights["conv"].shape) == (5, F)

    # Compare in [B, P, C_out] form
    x_patches = data.inputs["conv"].permute(0, 2, 1)  # [B, P, F]
    y_pred = x_patches @ data.weights["conv"].T  # [B, P, C_out]
    y_true = data.outputs["conv"].permute(0, 2, 1)  # [B, P, C_out]
    torch.testing.assert_close(y_pred, y_true, rtol=1e-4, atol=1e-5)


def test_activation_capture_conv1d_unfold_matches_conv(device):
    """
    Conv1d support: unfold mode should produce inputs [B*P, C_in*k] and
    outputs [B*P, C_out], matching the Conv1d forward when bias=False.
    """

    class SimpleConv1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(2, 3, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = SimpleConv1d().to(device)
    wrapper = ModelWrapper(model, tracked_layers=["conv"], preprocessing_mode="auto")
    service = ActivationCaptureService(wrapper, default_mode="unfold")

    B, C, L = 2, 2, 17
    x = torch.randn(B, C, L, device=device)

    data = service.capture(x, layers=["conv"], include_weights=True, preprocess=True)

    # Output length: floor((L + 2p - d*(k-1) - 1)/s + 1)
    k, s, p, d = 5, 2, 2, 1
    L_out = (L + 2 * p - d * (k - 1) - 1) // s + 1
    P = L_out
    F = C * k

    assert tuple(data.inputs["conv"].shape) == (B * P, F)
    assert tuple(data.outputs["conv"].shape) == (B * P, 3)
    assert tuple(data.weights["conv"].shape) == (3, F)

    pred = data.inputs["conv"] @ data.weights["conv"].T
    torch.testing.assert_close(pred, data.outputs["conv"], rtol=1e-4, atol=1e-5)
